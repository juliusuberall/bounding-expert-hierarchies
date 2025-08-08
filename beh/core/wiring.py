import beh.core.moe_benchmarking as moe_B
import beh.core.mlp_benchmarking as mlp_B

from beh.core.moe import export_moe
from beh.core.shared import batch_data

from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *
from beh.core.registry import *
from beh.core.train_moe import train_moe
from beh.core.train_mlp import train_mlp

def train_model(
    model_key : str,
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query : str,
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
            query = query,
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
            query = query,
            dimension = dimension
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

    # Batch data
    x_batches = batch_data(x, batch_size)

    # Get general configs
    model_type = configs[model_key]['type']

    if model_type == 'moe':
        reg = moe_B.register_accuracy(model_key, model, x_batches, y, reg, threshold)
        reg = moe_B.register_gating_confidence(model_key, model, x_batches, reg)
        reg = moe_B.register_all_expert_boundaries(model_key, model, x_batches, reg)
        reg = moe_B.register_inference_speed(model_key, model, x_batches, reg, configs, dimension)
        return reg
    
    elif model_type == 'mlp':
        reg = mlp_B.register_accuracy(model_key, model, x_batches, y, reg, threshold)
        reg = mlp_B.register_inference_speed(model_key, model, x_batches, reg, configs, dimension)
        return reg
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def save_model (
    model_key : str,
    configs : dict,
    model) -> str:
    '''
    Export the parameters of a model.
    \nReturns path to exported model file.
    '''

    # Get general configs
    model_type = configs[model_key]['type']

    # Not the cleanest way of delegating, but works for our case
    if model_type == 'moe':
        return export_moe(model, model_key)
    elif model_type == 'mlp':
        pass
    else:
        raise ValueError(f"Could not export the model. Unsupported model format.")