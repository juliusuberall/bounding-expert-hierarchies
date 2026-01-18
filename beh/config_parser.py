import argparse
import yaml

from beh.registry import *
from beh.core.params import count_parameter
from beh.core.shared import pe_dim
from beh.core.registry import gating_table

def parse_train():
    '''
    Retrieve command line arguments and model configuration yaml for invoking model training and evaluation.
    '''
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        required=True, 
        type = str,
        default = None,
        help ='Name of training data')
    parser.add_argument(
        "--query",
        required=True, 
        choices = ["point", "ray"],
        type = str,
        default = None,
        help = "Query type") 
    parser.add_argument(
        "--dim",
        required=True, 
        type = int,
        choices = [ 2, 3, 4, 9],
        default = None,
        help = "Query dimension")
    args = parser.parse_args()

    # Load model configurations / architectures
    with open(model_config_registry[args.dim], "r") as file:
        configs = yaml.safe_load(file)

    return args, configs

#------------------------------------------------------------------------------------

def parse_sample():
    '''
    Retrieve command line arguments for invoking data sampling.
    '''
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        required=True, 
        type = str,
        default = None,
        help ='Name of data to sample')
    parser.add_argument(
        '--query',
        required=True, 
        choices = ["point", "ray"],
        type = str,
        default = None,
        help ='Type of query to sample or pre-process')
    parser.add_argument(
        "--dim",
        required=True, 
        type = int,
        choices = [ 2, 3, 4, 9],
        default = None,
        help = "Query dimension")
    parser.add_argument(
        "--res",
        required=False, 
        type = int,
        default = None,
        help = "Dimension Resolution")
    parser.add_argument(
        "--strategy",
        required=False, 
        type = str,
        choices = [ 'random', 'grid'],
        default = None,
        help = "Sampling Strategy")
    parser.add_argument(
        "--start",
        required=False, 
        type = int,
        default = None,
        help = "Dataset Start Frame")
    parser.add_argument(
        "--end",
        required=False, 
        type = int,
        default = None,
        help = "Dataset End Frame")
    parser.add_argument(
        "--frame_rate",
        required=False, 
        type = int,
        default = None)
    
    return parser.parse_args()

#------------------------------------------------------------------------------------

def compute_config_model_sizes(dim : int, query_type : str):
    # Load model configurations / architectures
    with open(model_config_registry[dim], "r") as file:
        configs = yaml.safe_load(file)
    config_list = list(configs.keys())

    # Compute positional encoding dimensions
    query_dim = dim if query_type == "point" else dim * 2
    if dim == 4 and query_type == "ray" :
        query_dim -= 1
    pe_i = pe_dim(query_dim, configs['general']['pe_num_freq'])

    # Create caches
    bvh, mlp, moeg, moe = [], [], [], []

    # Count model parameters
    for key in config_list:
        if key == 'general' : continue
        match configs[key]['type']:
            case 'bvh':
                # Max depth = for each depth i we store 2^i bounding boxes with 2 n-D coordinates 
                depth = configs[key]['max_depth']
                p = 0
                for i in range(1, depth+1):
                    p += 2**i * (2 * dim)
                bvh.append(p)

            case 'mlp':
                arch = [pe_i] + configs[key]['hidden_layer'] + [1]
                p = count_parameter(arch)
                mlp.append(p)
            case 'moeg':
                nex = configs[key]['grid_dim'] ** dim
                e_arch = [pe_i] + configs[key]['expert_hidden_layer'] + [1]
                p = nex * count_parameter(e_arch)
                moeg.append(p)
            case 'moe':
                nex = configs[key]['nex']
                g_arch = [gating_table[query_type][query_dim]] + configs[key]['gate_hidden_layer'] + [nex]
                e_arch = [pe_i] + configs[key]['expert_hidden_layer'] + [1]
                p = count_parameter(g_arch) + nex * count_parameter(e_arch)
                moe.append(p)

    print('Model Parameter (S,M,L)')
    print(f'BVH : {bvh}')
    print(f'MLP : {mlp}')
    print(f'MOEG: {moeg}')
    print(f'MOE : {moe}')