from os import nice
from pathlib import Path
import logging
logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)
import os
import torch
import numpy as np
from openfold.data import data_pipeline
from explainable_fold.utils.functions import init_name_dicts
from run_pretrained_openfold import load_models_from_command_line
from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline
from openfold.data import templates, parsers, mmcif_parsing


def get_seq_from_vec_sub(clampled_delta):
    i_to_c_dict, i_to_fc_fict, c_to_i_dict, fc_to_i_dict = init_name_dicts()
    clampled_delta = np.array(clampled_delta)
    clampled_delta[:, 0] = 0
    clampled_delta[:, -1] = 0
    int_ids = np.argmax(clampled_delta, axis=1)
    string = []
    for int_i in int_ids:
        string.append(i_to_fc_fict[int_i])
    string = ''.join(string)
    return [string]

def get_seq_from_vec_del(target_feat):
    i_to_c_dict, i_to_fc_fict, c_to_i_dict, fc_to_i_dict = init_name_dicts()
    target_feat = np.array(target_feat)
    int_ids = np.argmax(target_feat, axis=1)
    string = []
    for int_i in int_ids:
        string.append(i_to_fc_fict[int_i[0]])
    string = ''.join(string)
    return string


def generate_feature_dict(
    tag,
    seq,
    alignment_dir,
    data_processor,
    args,
    replace=False
):
    if os.path.exists(os.path.join(alignment_dir, 'complete.txt')) and not replace: 
        print('use pre-computed MSAs')
    else:
        compute_msas(tag, seq, alignment_dir, args, replace=replace)
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag[0]}\n{seq}")
    feature_dict = data_processor.process_fasta(
        fasta_path=tmp_fasta_path, alignment_dir=alignment_dir
    )
    os.remove(tmp_fasta_path)
    return feature_dict

def compute_msas(tag, seq, alignment_dir, args, replace=False):
    """
    generate MSAs, no matter pre-computed alignments exist or not.
    tag: sequence name
    seq: primary sequence
    alignment_dir: mas directory
    args: args
    replace: if pre-computed MSAs exist, choose to replace them or not.
    """

    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")
    logger.info(f"Generating alignments for {tag}...")
    if os.path.exists(os.path.join(alignment_dir, 'complete.txt')) and not replace: 
        print('use pre-computed MSAs')
    else:
        Path(alignment_dir).mkdir(parents=True, exist_ok=True)  # set save directory
        alignment_runner = data_pipeline.AlignmentRunner(
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            hhsearch_binary_path=args.hhsearch_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            bfd_database_path=args.bfd_database_path,
            uniclust30_database_path=args.uniclust30_database_path,
            pdb70_database_path=args.pdb70_database_path,
            no_cpus=args.cpus,
            )
        alignment_runner.run(tmp_fasta_path, alignment_dir)
        with open(os.path.join(alignment_dir, 'complete.txt'), 'w') as f:
            f.write('complete')
    os.remove(tmp_fasta_path)
    return True


def list_files_with_extensions(dir, extensions, pool):
    fasta_files = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name.endswith(extensions):
                if pool is None:  # explain all proteins
                    fasta_files.append(os.path.join(path, name))
                else:
                    if name.split('.')[0][-4:] in pool:
                        fasta_files.append(os.path.join(path, name))
    return fasta_files


def flex_alignment():
    "align two protein sequences with minimum TM scores"
    return True


def save_seq(seq, seq_path):
    with open(seq_path, 'w') as f:
        f.write(seq)
    return True


def load_seq(seq_path):
    seq = ""
    with open(seq_path, 'r') as f:
        seq = f.read()
    return seq


def model_initialization(args):
    # initialize feature processor, data processor, and alphafold model
    config = model_config(args.config_preset)  # model config
    # processor
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    template_featurizer = None
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )  # JT: Init feature model
    # init model
    model_generator = load_models_from_command_line(
        config,
        args.model_device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir)
    model, output_directory = next(model_generator)
    # model.train()
    for param in model.parameters():
        param.requires_grad = False
    return feature_processor, data_processor, model


def predict_result(tag, seq, alignment_dir, chunk_id, phase_id, data_processor, feature_processor, model, args):
    """
    predict results with given chunk id and phase id
    """

    feature_dict = generate_feature_dict(
                tags=tag,
                seqs=seq,
                alignment_dir=alignment_dir,
                chunk_id=chunk_id,
                phase_id=phase_id,
                data_processor=data_processor,
                args=args,)
    # print(init_feature_dict['aatype'][:10])

    init_processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict')

    init_processed_feature_dict = {k:torch.as_tensor(v, device=args.model_device)
        for k,v in init_processed_feature_dict.items()}  # put features into device
    init_result = model(init_processed_feature_dict)
    return init_result


def parse_msas(msa_dir):
    for f in os.listdir(msa_dir):
        path = os.path.join(msa_dir, f)
        ext = os.path.splitext(f)[-1]
        if(ext == ".a3m"):
            with open(path, "r") as fp:
                msa, deletion_matrix = parsers.parse_a3m(fp.read())
            data = {"msa": msa, "deletion_matrix": deletion_matrix}
        elif(ext == ".sto"):
            with open(path, "r") as fp:
                msa, deletion_matrix, _ = parsers.parse_stockholm(
                    fp.read()
                )
            data = {"msa": msa, "deletion_matrix": deletion_matrix}
        else:
            continue
    return msa