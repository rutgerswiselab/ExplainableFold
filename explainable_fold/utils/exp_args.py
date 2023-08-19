import argparse
import os

def add_exp_args(parser: argparse.ArgumentParser):
    """
    some common arguments
    """
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noconf', action='store_true')
    parser.add_argument('--alp', type=float, default=0.2, help="margin value")
    parser.add_argument('--steps', type=int, default=100, help="number of training steps")
    parser.add_argument('--loss_type', type=str, default='tm', help="align with <rmsd> or <tm>")
    parser.add_argument('--opt_type', type=str, default='min', help="choose from <min> and <max>")
    parser.add_argument('--delta_init', type=str, default='rand', help="delta initialization, choose from <rand>, <equal>")
    parser.add_argument('--max_l', type=int, default=384, help="max lenth of sub sequences, too long proteins may not fit into the model")
    parser.add_argument("--fasta_dir", type=str, help="path to directory containing FASTA files, one sequence per file")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="""Name of the directory in which to output the prediction""",)
    parser.add_argument("--config_preset", type=str, default="model_3_ptm", help="""Name of a model config preset defined in openfold/config.py""")
    parser.add_argument("--openfold_checkpoint_path", type=str, default=None, help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
    checkpoint directory or a .pt file""")
    parser.add_argument("--jax_param_path", type=str, default="openfold/resources/params/params_model_3_ptm.npz",
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument("--model_device", type=str, default="cpu", help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate in optimization")
    parser.add_argument("--num_phase", type=int, default=3, help="number of re-alignment")
    parser.add_argument("--leaky", type=float, default=0.1)
    parser.add_argument("--lam", dest="lam", type=float, default=1, help="tradeoff")
    parser.add_argument('--pool', nargs='+', help="the name of proteins to be explained")
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )