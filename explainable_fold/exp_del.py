import argparse
import os
import torch
import numpy as np
from pathlib import Path
from openfold.config import model_config
from scripts.utils import add_data_args
from openfold.data import data_pipeline, feature_pipeline
from run_pretrained_openfold import load_models_from_command_line, precompute_alignments, generate_feature_dict
from openfold.utils.script_utils import parse_fasta, run_model

from explainable_fold.utils.functions import set_seed, get_cuda_info
from explainable_fold.utils.exp_utils import compute_msas, list_files_with_extensions
from explainable_fold.models.exp_models_del import DelModel
from explainable_fold.utils.exp_args import add_exp_args

torch.set_grad_enabled(True)

def del_main(args):
    if args.seed is not None:
        set_seed(args.seed)
    output_dir = os.path.join(args.output_dir, args.opt_type + "_del")
    Path(output_dir).mkdir(parents=True, exist_ok=True)  # set save directory
    config = model_config(args.config_preset)  # model config

    # processor
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    template_featurizer = None
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    ) 
    model_generator = load_models_from_command_line(
        config,
        args.model_device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir)
    model, output_directory = next(model_generator)
    model.bfloat16()

    for param in model.parameters():
        param.requires_grad = False

    # generate fastas
    test_seqs = []
    fasta_files = list_files_with_extensions(args.fasta_dir, 
        (".fasta", ".fa"), 
        args.pool)

    for f_file in fasta_files:
        with open(f_file, "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        for i in range(len(tags)):
            seq_list = list(seqs[i])
            sub_seqs = []
            for j in range(0, len(seq_list), args.max_l):
                sub_seqs.append([''.join(seq_list[j: j+args.max_l])])
            test_seqs.append({'tag': tags[i], 'seq': seqs[i], 'sub_seqs': sub_seqs})

    exp_model = DelModel(
        base_model=model, 
        feature_processor=feature_processor, 
        data_processor=data_processor, 
        test_seqs = test_seqs,
        output_dir=output_dir,
        args=args
    )

    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    exp_model.generate_explanations()
    return True


if __name__ == "__main__":
    # set manual seed
    parser = argparse.ArgumentParser()
    add_exp_args(parser)
    add_data_args(parser)  # for msa
    args = parser.parse_args()      
    del_main(args)