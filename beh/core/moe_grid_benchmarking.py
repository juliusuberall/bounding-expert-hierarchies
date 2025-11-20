import jax
import jax.numpy as jnp
import numpy as np

from beh.core.shared import *
from beh.core.moe_grid import batch_query_moe_grid, batch_query_moe_grid_OOM, moe_grid_forward
from beh.core.registry import *
from beh.core.benchmarking import *

def register_accuracy(
        model_key : str,
        moe : dict,
        x_batches : list,
        x_aa : list,
        y : jax.Array,
        reg : CoreRegistry,
        threshold : float,
        dimension : int,
        grid_dim : int
    ):
    '''
    Computes and registers the accuracy with the provided training data batches.
    \nBased on those model outputs conservativness is computed and registered with false-negative,FN and false-positive,FP rates in % of max FN or FP that could be reached.
    '''
    # MoE Grid inference
    yp, idx, yp_raw = batch_query_moe_grid(x_batches, moe, dimension, grid_dim)
    fn, fp = get_fn_fp_rate(yp, y, threshold=threshold)

    if x_aa != None : 
        # 2D - Query in higher res for AA while avoiding OOM
        yp_aa = batch_query_moe_grid_OOM(x_aa, moe, 50, dimension, grid_dim)
        reg.add(model_key + core_keys['aa_y_prediciton_key'], yp_aa)
    
    # Save results
    reg.add(model_key + core_keys['y_prediciton_key'], yp)
    reg.add(model_key + core_keys['y_prediciton_RAW_key'], yp_raw)
    reg.add(model_key + core_keys['fn_key'], fn)
    reg.add(model_key + core_keys['fp_key'], fp)
    reg.add(model_key + core_keys['gate_top1_activation_key'], idx)

    print(f"FN: {float(fn):05f}")
    print(f"FP: {float(fp):05f}")

    return reg