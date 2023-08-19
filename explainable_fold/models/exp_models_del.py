import copy
import torch
from pathlib import Path
import numpy as np
print('torch version: ', torch.__version__)
import os
import tqdm
# from run_pretrained_openfold import parse_fasta
from explainable_fold.utils.structure_utils import compute_transformation, transform, compute_tm_score
from explainable_fold.utils.functions import get_cuda_info, torch_32_to_16
from explainable_fold.utils.exp_utils import compute_msas, generate_feature_dict, get_seq_from_vec_del, save_seq
# from torch.utils.checkpoint import checkpoint_sequential


class DelModel(torch.nn.Module):
    def __init__(self, base_model, feature_processor, data_processor, test_seqs, output_dir, args):
        super(DelModel, self).__init__()
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
                if chunk_len < 80:  # only consider protein lenth > 80. Refer to alpha fold paper
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

                # for k, v in init_processed_feature_dict.items():
                #     if v.dtype == torch.float32:
                #         init_processed_feature_dict[k] = v.type(torch.bfloat16)
                init_processed_feat_dict = torch_32_to_16(init_processed_feat_dict)
                init_processed_feat_dict = {k:torch.as_tensor(v, device=self.args.model_device)
                    for k,v in init_processed_feat_dict.items()} 

                init_result = self.base_model(init_processed_feat_dict)
                init_pos = init_result['final_atom_positions'][:, 1, :]  # only alpha-carbon
                init_conf = torch.where(init_result['plddt']>70)[0]
                conf_percent = len(init_conf) / chunk_len * 100
                print('conf_percentage: ', conf_percent)

                with open(os.path.join(local_output_dir, 'conf_percent.txt'), 'w') as f:
                    f.write(str(conf_percent))
                
                if conf_percent < 30:  # skip the chaines that alphafold is not confident enough
                    print("this chain is not suitable: ", tag, c_id)
                    continue

                # init delta
                delta_shape = init_processed_feat_dict['target_feat'].shape
                delta = torch.ones(delta_shape[0]) * 0.1

                for phase in range(self.args.num_phase):
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

                    delta, target_feat = self.explain(
                                                chunk_id=c_id, 
                                                phase=phase, 
                                                seq=cur_chunk, 
                                                processed_feature_dict=processed_feat_dict, 
                                                init_processed_feature_dict=init_processed_feat_dict, 
                                                delta=delta, 
                                                init_pos=init_pos, 
                                                init_conf=init_conf,
                                                local_output_dir=local_output_dir)
                    # change chunk, MSAs and go to next phase
                    cur_chunk = get_seq_from_vec_del(target_feat.detach().to('cpu').numpy())
                    print('cur chunk: ', cur_chunk)
                    with open(os.path.join(local_output_dir, 'chunks', f'chunk_{c_id}_phase_{phase+1}.txt'), 'w') as f:
                        f.write(cur_chunk)

                    # compute_msas(tags=tags, 
                    #             seqs=cur_chunk, 
                    #             phase=phase+1, 
                    #             local_out_path=local_out_path, 
                    #             args=self.args, 
                    #             chunk_id=chunk_id)  # compute MSA and put them into target folder
        return True
    
    # def generate_mask(self, shape):
    #     mask = torch.ones(shape[0], shape[1])
    #     mask[:, 0] = 0
    #     mask[:, -1] = 0
    #     return mask

    def explain(self, chunk_id, phase, seq, processed_feature_dict, init_processed_feature_dict, delta, init_pos, init_conf, local_output_dir):
        exp_generator = EXPGenerator(
            seq,
            self.base_model,
            init_processed_feature_dict,
            delta,
            init_pos,
            init_conf,
            self.args
        ).bfloat16().to(self.args.model_device)
        # optimizer = torch.optim.Adam([exp_generator.delta], lr=self.args.lr, weight_decay=0.00001)
        optimizer = torch.optim.Adam([exp_generator.delta], lr=self.args.lr, weight_decay=0.00001)
        # exp_generator.train()
        losses = []
        optimized_loss = np.inf
        if self.args.opt_type == 'min':
            optimized_dist_l = -1
        else:
            optimized_dist_l = 0
        
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
            if self.args.opt_type == 'min':
                if dist_l > optimized_dist_l:
                    optimized_delta = exp_generator.delta
                    optimized_dist_l = dist_l
            else:
                if loss < optimized_loss:
                    optimized_delta = exp_generator.delta
                    optimized_loss = loss
        np.array(losses).dump(os.path.join(local_output_dir, 'losses', f'losses_chunk_{chunk_id}_phase_{phase}.pickle'))
        clamped_delta = exp_generator.clamp_delta(optimized_delta)
        target_feat = exp_generator.get_target_feat(optimized_delta)
        return clamped_delta, target_feat


class EXPGenerator(torch.nn.Module):
    def __init__(self, seq, base_model, init_feature_dict, delta, init_pos, init_conf, args):
        super(EXPGenerator, self).__init__()
        self.args = args
        self.base_model = base_model
        self.delta = torch.nn.Parameter(torch.clone(delta))
        self.init_feature_dict = init_feature_dict.copy()
        self.init_pos = init_pos
        self.init_conf = init_conf

    def clamp_delta(self, delta):
        delta = torch.clamp(delta, 0, 1)
        return delta
    
    def get_target_feat(self, delta):
        clamped_delta = self.clamp_delta(delta)
        unk_vec = torch.zeros(self.init_feature_dict['target_feat'].shape).to(self.args.model_device)
        unk_vec[:, 21, :] = 1
        input_delta = torch.unsqueeze(clamped_delta, dim=-1).repeat(1, 22)
        input_delta = torch.unsqueeze(input_delta, dim=-1).repeat(1, 1, 4)
        target_feat = self.init_feature_dict['target_feat'] * (1 - input_delta) + unk_vec * input_delta
        return target_feat
    
    def forward(self, processed_feature_dict):
        target_feat = self.get_target_feat(self.delta)
        processed_feature_dict['target_feat'] = target_feat
        out = self.base_model(processed_feature_dict)
        return out
    
    def loss_min(self, result):
        pos_2 = result['final_atom_positions'][:, 1, :]
        T = compute_transformation(self.init_pos[self.init_conf].to('cpu').numpy(), pos_2[self.init_conf].detach().to('cpu').numpy())
        pos_1_trans = transform(self.init_pos, torch.Tensor(T).to(self.args.model_device))

        # obsolete
        # if self.args.skip_x:
        #     structure_loss_ids = self.exclude_x_ids(self.init_conf.tolist(), self.x_ids.tolist())
        #     l1_loss_ids = self.exclude_x_ids([i for i in range(len(self.delta))], self.x_ids.tolist())
        # else:
        #     structure_loss_ids = self.init_conf
        #     l1_loss_ids = torch.tensor([i for i in range(len(self.delta))]).long().to(self.args.model_device)

        structure_loss_ids = self.init_conf
        l1_loss_ids = torch.tensor([i for i in range(len(self.delta))]).long().to(self.args.model_device)

        if self.args.loss_type == "rmsd":
            if self.args.noconf:
                dist_loss = torch.linalg.norm(pos_1_trans - pos_2) / len(pos_1_trans)
            else:
                dist_loss = torch.linalg.norm(pos_1_trans[structure_loss_ids] - pos_2[structure_loss_ids]) / len(structure_loss_ids)
        elif self.args.loss_type == "tm":
            # dist_loss = compute_tm_score_tensor(pos_1_trans, pos_2)
            if self.args.noconf:
                dist_loss = - torch.nn.functional.leaky_relu(compute_tm_score(pos_1_trans, pos_2) - 0.5 + self.args.alp, self.args.leaky)
            else:
                tm = compute_tm_score(pos_1_trans[structure_loss_ids], pos_2[structure_loss_ids])
                dist_loss = - torch.nn.functional.leaky_relu(tm - 0.5 + self.args.alp, self.args.leaky)

        clamped_delta = self.clamp_delta(self.delta)
        l1 = torch.linalg.norm(clamped_delta[l1_loss_ids], ord=1) / len(clamped_delta[l1_loss_ids])
        loss = self.args.lam * l1 - dist_loss
        return l1, dist_loss, loss

    def loss_max(self, result):
        pos_2 = result['final_atom_positions'][:, 1, :]
        T = compute_transformation(self.init_pos[self.init_conf].to('cpu').numpy(), pos_2[self.init_conf].detach().to('cpu').numpy())
        pos_1_trans = transform(self.init_pos, torch.Tensor(T).to(self.args.model_device))

        # if self.args.skip_x:
        #     structure_loss_ids = self.exclude_x_ids(self.init_conf.tolist(), self.x_ids.tolist())
        #     l1_loss_ids = self.exclude_x_ids([i for i in range(len(self.delta))], self.x_ids.tolist())
        # else:
        #     structure_loss_ids = self.init_conf
        #     l1_loss_ids = torch.tensor([i for i in range(len(self.delta))]).long().to(self.args.model_device)

        structure_loss_ids = self.init_conf
        l1_loss_ids = torch.tensor([i for i in range(len(self.delta))]).long().to(self.args.model_device)
        if self.args.loss_type == "rmsd":
            if self.args.noconf:
                dist_loss = torch.linalg.norm(pos_1_trans - pos_2) / len(pos_1_trans)
            else:
                dist_loss = torch.linalg.norm(pos_1_trans[structure_loss_ids] - pos_2[structure_loss_ids]) / len(structure_loss_ids)
        elif self.args.loss_type == "tm":
            if self.args.noconf:
                dist_loss = torch.nn.functional.leaky_relu(- (compute_tm_score(pos_1_trans, pos_2) - 0.5 - self.args.alp), self.args.leaky)
            else:
                tm = compute_tm_score(pos_1_trans[structure_loss_ids], pos_2[structure_loss_ids])
                dist_loss = torch.nn.functional.leaky_relu(- (tm - 0.5 - self.args.alp), self.args.leaky)
        # compute change
        clamped_delta = self.clamp_delta(self.delta)
        l1 = torch.linalg.norm(clamped_delta[l1_loss_ids], ord=1) / len(clamped_delta[l1_loss_ids])
        loss = - self.args.lam * l1 + dist_loss
        return l1, dist_loss, loss
    
    def exclude_x_ids(self, ids_1, ids_2):
        """
        exclude ids_2 from ids_1
        """
        ids = list(set(ids_1) - set(ids_2))
        ids = torch.tensor(ids).long().to(self.args.model_device)
        return ids