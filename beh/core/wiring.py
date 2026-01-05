import beh.core.moe_benchmarking as moe_B
import beh.core.moe_grid_benchmarking as moeG_B
import beh.core.mlp_benchmarking as mlp_B
import beh.core.bvh_benchmarking as bvh_B

from beh.core.moe import export_moe

from beh.core.shared import batch_data
from beh.core.registry import *
from beh.core.bvh import build_bvh, shrink_to_nex_power_of_two
from beh.core.train_moe import train_moe
from beh.core.train_moe_grid import train_moe_grid
from beh.core.train_mlp import train_mlp
from beh.core.shared import create_queries

from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *

def train_model(
    model_key : str,
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query_dim : int,
    configs : dict,
    dimension : int):

    # Get general configs
    model_type = configs[model_key]['type']

    if model_type == 'moe':
        return train_moe(
            model_key = model_key,
            key = key,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            query_dim = query_dim,
            dimension = dimension
        )
    elif model_type == 'moe_grid':
        return train_moe_grid(
            model_key = model_key,
            key = key,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            query_dim = query_dim,
            dimension = dimension
        )
    elif model_type == 'mlp':
        return train_mlp(
            model_key = model_key,
            key = key,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            query_dim = query_dim,
            dimension = dimension
        )
    elif model_type == 'bvh':
        # Filter out data that is labeled as occupied or hit 
        # and account for balanced bvh tree by reducing amount of
        # points to construct bvh with
        x = x[y > 0]
        x = shrink_to_nex_power_of_two(x)
        return build_bvh(
            positions = x,
            max_depth = configs[model_key]['max_depth'],
            reg = reg
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# An orchestration function that loads the correct data 
def get_benchmarks(
    model_key : str,
    model ,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    configs : dict,
    dimension : int) -> CoreRegistry :
    '''
    Run all benchmarks and compute all results with the model.
    \nDelegates to corresponding data dimensionality sub-routine.
    '''

    # Get general configs
    batch_size = configs['general']['batch_size']
    threshold = configs['general']['boundary_threshold']
    infB_qsize = configs['general']['inf_bench_query_size']
    infB_reps = configs['general']['inf_bench_repitions'] 
    infB_batch_size = configs['general']['infB_batch_size']
    inf = configs['general']['inf_benchmark'] 

    # Dimensions
    img_dim_0, img_dim_1, _ = reg.get(core_keys['data_size_key'])

    # Batch data
    x_batches = batch_data(x, batch_size)

    # Create queries for anti-aliasing binary classification plots in 2D
    x_aa, s = None, None
    if dimension == 2:
        s = configs['general']['aa_scaling'] 
        x_aa = batch_data(create_queries(width=img_dim_0*s, height=img_dim_1*s), batch_size=2048)

    # Get general configs
    model_type = configs[model_key]['type']

    if model_type == 'moe':
        reg = moe_B.register_accuracy(model_key, model, x_batches, x_aa, y, reg, threshold)
        reg = moe_B.register_gating_confidence(model_key, model, x_batches, x_aa, reg)
        reg = moe_B.register_inference_speed(inf, model_key, model, x, reg, dimension, infB_reps, infB_qsize, infB_batch_size)
        return reg
    
    elif model_type == 'moe_grid':
        reg = moeG_B.register_accuracy(model_key, model, x_batches, x_aa, y, reg, threshold)
        reg = moeG_B.register_inference_speed(inf, model_key, model, x, reg, dimension, infB_reps, infB_qsize, infB_batch_size)
        return reg

    elif model_type == 'mlp':
        reg = mlp_B.register_accuracy(model_key, model, x_batches, x_aa, y, reg, threshold)
        reg = mlp_B.register_inference_speed(inf, model_key, model, x, reg, dimension, infB_reps, infB_qsize, infB_batch_size)
        return reg
    
    elif model_type == 'bvh':
        reg = bvh_B.register_accuracy(model_key, model, x_batches, x_aa, y, reg, threshold)
        reg = bvh_B.register_inference_speed(inf, model_key, model, x, reg, dimension, infB_reps, infB_qsize, infB_batch_size)
        return reg
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")