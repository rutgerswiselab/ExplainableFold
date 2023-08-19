import copy
import torch
from pathlib import Path
import numpy as np
print('torch version: ', torch.__version__)
import os
import tqdm
from run_pretrained_openfold import parse_fasta
from explainable_fold.utils.structure_utils import compute_transformation, transform, compute_tm_score
from explainable_fold.utils.functions import get_cuda_info, torch_32_to_16
from explainable_fold.utils.exp_utils import compute_msas, generate_feature_dict, get_seq_from_vec_sub, save_seq


class SubModel(torch.nn.Module):
    def __init__(self, base_model, feature_processor, data_processor, test_seqs, output_dir, args):
        super(SubModel, self).__init__()
        self.base_model = base_model
        self.feature_processor = feature_processor
        self.data_processor = data_processor
        self.test_seqs = test_seqs
        # self.alignment_dir = alignment_dir
        self.output_dir = output_dir
        self.args = args

    def generate_explanations(self):
        for test_seq in self.test_seqs:
            tag = copy.copy(test_seq['tag'])
            seq = copy.copy(test_seq['seq'])
            local_output_dir = os.path.join(self.output_dir, tag)
            print('generate explanation for sequence: ', tag)
            Path(os.path.join(local_output_dir, 'chunks')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(local_output_dir, 'losses')).mkdir(parents=True, exist_ok=True)

            chunks = copy.copy(test_seq['sub_seqs'])
            # explain each chunks separately
            for c_id in range(len(chunks)):
                cur_chunk = chunks[c_id][0]
                chunk_len = len(cur_chunk)
                print('length: ', chunk_len)
                if chunk_len < 80:
                    continue
                # save original chunk first
                save_seq(str(cur_chunk), os.path.join(local_output_dir, 'chunks', f'chunk_{c_id}_phase_0.txt'))

                init_feat_dict = generate_feature_dict(
                            tag=tag,
                            seq=cur_chunk,
                            alignment_dir=os.path.join('explainable_fold', 'alignments', tag, 'alignment', f"chunk_{c_id}, phase_0"),
                            data_processor=self.data_processor,
                            args=self.args,)
                init_processed_feat_dict = self.feature_processor.process_features(
                            init_feat_dict, mode='predict')

                init_processed_feat_dict = torch_32_to_16(init_processed_feat_dict)
                init_processed_feat_dict = {k:torch.as_tensor(v, device=self.args.model_device)
                    for k,v in init_processed_feat_dict.items()} 
                init_result = self.base_model(init_processed_feat_dict)
                init_pos = init_result['final_atom_positions'][:, 1, :]
                init_conf = torch.where(init_result['plddt']>70)[0]  # the target chunk
                conf_percent = len(init_conf) / chunk_len * 100
                print('conf_percentage: ', conf_percent)

                with open(os.path.join(local_output_dir, 'conf_percent.txt'), 'w') as f:
                    f.write(str(conf_percent))
                
                if conf_percent < 30:  # skip the chaines that alphafold is not confident enough
                    print("this chain is not suitable: ", tag, c_id)
                    continue

                # init delta
                delta_shape = init_processed_feat_dict['target_feat'].shape
                if self.args.delta_init == 'rand':
                    cur_delta = torch.FloatTensor(delta_shape[0], delta_shape[1]).uniform_(0, 1)
                else:
                    cur_delta = torch.clone(init_processed_feat_dict['target_feat'][:, :, 0])
                    cur_delta = torch.where(cur_delta==0, -2.0, 2.0)

                delta_mask = self.generate_mask(delta_shape).to(self.args.model_device)
                phase_num = 4

                for phase in range(self.args.num_phase):  # each chunk has 3 phase
                    print('phase: ', phase)
                    if phase == 0:
                        feat_dict = copy.copy(init_feat_dict)
                    else:
                        feat_dict = generate_feature_dict(
                            tag=tag,
                            seq=cur_chunk,
                            alignment_dir=os.path.join(local_output_dir, 'alignment', f"chunk_{c_id}_phase_{phase}"),
                            data_processor=self.data_processor,
                            args=self.args)
                    processed_feat_dict = self.feature_processor.process_features(
                        feat_dict, mode='predict')
                    processed_feat_dict = torch_32_to_16(processed_feat_dict)
                    processed_feat_dict = {k:torch.as_tensor(v, device=self.args.model_device) 
                        for k,v in processed_feat_dict.items()}  # put features into device
                    
                    cur_delta, clampled_delta = self.explain(
                                                chunk_id=c_id, 
                                                phase=phase, 
                                                seq=cur_chunk, 
                                                processed_feature_dict=processed_feat_dict, 
                                                init_processed_feature_dict=init_processed_feat_dict, 
                                                delta=cur_delta, 
                                                delta_mask=delta_mask, 
                                                init_pos=init_pos, 
                                                init_conf=init_conf, 
                                                local_output_dir=local_output_dir)
                    
                    # change chunk, MSAs and go to next phase
                    cur_chunk = get_seq_from_vec_sub(clampled_delta.type(torch.float32).detach().to('cpu').numpy())
                    with open(os.path.join(local_output_dir, 'chunks', f'chunk_{c_id}_phase_{phase+1}.txt'), 'w') as f:
                        f.write(cur_chunk[0])

                    # compute_msas(tags=tags, 
                    #             seqs=cur_chunk, 
                    #             phase=phase+1, 
                    #             local_out_path=local_out_path, 
                    #             args=self.args, 
                    #             chunk_id=chunk_id)  # compute MSA and put them into target folder
        return True
    
    def generate_mask(self, shape):
        mask = torch.ones(shape[0], shape[1])
        mask[:, 0] = 0
        mask[:, -1] = 0
        return mask

    def explain(self, chunk_id, phase, seq, processed_feature_dict, init_processed_feature_dict, delta, delta_mask, init_pos, init_conf, local_output_dir):
        exp_generator = EXPGenerator(
            seq,
            self.base_model,
            init_processed_feature_dict,
            delta,
            delta_mask,
            init_pos,
            init_conf,
            self.args
        ).bfloat16().to(self.args.model_device)

        optimizer = torch.optim.Adam([exp_generator.delta], lr=self.args.lr, weight_decay=0.00001)
        # exp_generator.train()
        losses = []
        optimized_loss = np.inf
        for step in tqdm.trange(self.args.steps):
            result = exp_generator(processed_feature_dict)
            if self.args.opt_type == 'min':
                l1, dist_l, loss = exp_generator.loss_min(result)
            elif self.args.opt_type == 'max':
                l1, dist_l, loss = exp_generator.loss_max(result)
            # print('step: ', step, '  l1:  ', self.args.lam * l1, '  dist_l: ', dist_l, '  loss: ', loss)
            losses.append([l1.type(torch.float32).detach().to('cpu').numpy(), 
                            dist_l.type(torch.float32).detach().to('cpu').numpy(), 
                            loss.type(torch.float32).detach().to('cpu').numpy()])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < optimized_loss:
                optimized_delta = exp_generator.delta
                optimized_loss = loss
        np.array(losses).dump(os.path.join(local_output_dir, 'losses', f'losses_chunk_{chunk_id}_phase_{phase}.pickle'))
        clamped_delta = exp_generator.clamp_delta(optimized_delta)
        # map mask to feature
        return optimized_delta, clamped_delta



class EXPGenerator(torch.nn.Module):
    def __init__(self, seq, base_model, init_feature_dict, delta, delta_mask, init_pos, init_conf, args):
        super(EXPGenerator, self).__init__()
        self.args = args
        self.base_model = base_model
        self.delta = torch.nn.Parameter(torch.clone(delta))
        self.delta_mask = delta_mask
        self.init_feature_dict = init_feature_dict.copy()
        self.init_pos = init_pos
        self.init_conf = init_conf

    def clamp_delta(self, delta):
        clamped_delta = torch.nn.functional.softmax(delta, dim=1)
        masked_delta = clamped_delta * self.delta_mask
        masked_delta = torch.nn.functional.normalize(masked_delta, p=1, dim=1)
        return masked_delta
    
    def forward(self, processed_feature_dict):
        clamped_delta = self.clamp_delta(self.delta)
        input_delta = torch.unsqueeze(clamped_delta, dim=-1).repeat(1, 1, 4)
        processed_feature_dict['target_feat'] = input_delta
        out = self.base_model(processed_feature_dict)
        return out
    
    def loss_min(self, result):
        pos_2 = result['final_atom_positions'][:, 1, :]
        T = compute_transformation(self.init_pos[self.init_conf].to('cpu').numpy(), pos_2[self.init_conf].detach().to('cpu').numpy())
        pos_1_trans = transform(self.init_pos, torch.Tensor(T).to(self.args.model_device))

        if self.args.loss_type == "rmsd":
            if self.args.noconf:
                dist_loss = torch.linalg.norm(pos_1_trans - pos_2) / len(pos_1_trans)
            else:
                dist_loss = torch.linalg.norm(pos_1_trans[self.init_conf] - pos_2[self.init_conf]) / len(self.init_conf)
        elif self.args.loss_type == "tm":
            if self.args.noconf:
                dist_loss = - torch.nn.functional.leaky_relu(compute_tm_score(pos_1_trans, pos_2) - 0.5 + self.args.alp, self.args.leaky)
            else:
                tm = compute_tm_score(pos_1_trans[self.init_conf], pos_2[self.init_conf])
                dist_loss = - torch.nn.functional.leaky_relu(tm - 0.5 + self.args.alp, self.args.leaky)
        # compute change
        clamped_delta = self.clamp_delta(self.delta)
        input_delta = torch.unsqueeze(clamped_delta, dim=-1).repeat(1, 1, 4)
        l1 = torch.linalg.norm((self.init_feature_dict['target_feat'].flatten() - input_delta.flatten()), ord=1) / (4 * len(input_delta))
        loss = self.args.lam * l1 - dist_loss
        return l1, dist_loss, loss

    def loss_max(self, result):
        pos_2 = result['final_atom_positions'][:, 1, :]
        T = compute_transformation(self.init_pos[self.init_conf].to('cpu').numpy(), pos_2[self.init_conf].detach().to('cpu').numpy())
        pos_1_trans = transform(self.init_pos, torch.Tensor(T).to(self.args.model_device))
        if self.args.loss_type == "rmsd":
            if self.args.noconf:
                dist_loss = torch.linalg.norm(pos_1_trans - pos_2) / len(pos_1_trans)
            else:
                dist_loss = torch.linalg.norm(pos_1_trans[self.init_conf] - pos_2[self.init_conf]) / len(self.init_conf)
        elif self.args.loss_type == "tm":
            if self.args.noconf:
                dist_loss = torch.nn.functional.leaky_relu(- (compute_tm_score(pos_1_trans, pos_2) - 0.5 - self.args.alp), self.args.leaky)
            else:
                tm = compute_tm_score(pos_1_trans[self.init_conf], pos_2[self.init_conf])
                dist_loss = torch.nn.functional.leaky_relu(- (tm - 0.5 - self.args.alp), self.args.leaky)
        # compute change
        clamped_delta = self.clamp_delta(self.delta)
        input_delta = torch.unsqueeze(clamped_delta, dim=-1).repeat(1, 1, 4)
        l1 = torch.linalg.norm((self.init_feature_dict['target_feat'].flatten() - input_delta.flatten()), ord=1) / (4 * len(input_delta))
        loss = - self.args.lam * l1 + dist_loss
        return l1, dist_loss, loss