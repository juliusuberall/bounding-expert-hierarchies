import jax
import jax.numpy as jnp
import numpy as np

from beh.core.shared import *
from beh.core.moe_grid import batch_query_moe_grid, batch_query_moe_grid_OOM, moe_grid_forward_INF
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
    ):
    '''
    Computes and registers the accuracy with the provided training data batches.
    \nBased on those model outputs conservativness is computed and registered with false-negative,FN and false-positive,FP rates in % of max FN or FP that could be reached.
    '''
    # MoE Grid inference
    yp, idx, yp_raw = batch_query_moe_grid(x_batches, moe)
    fn, fp = get_fn_fp_rate(yp, y, threshold=threshold)

    if x_aa != None : 
        # 2D - Query in higher res for AA while avoiding OOM
        yp_aa = batch_query_moe_grid_OOM(x_aa, moe, 50)
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

def register_inference_speed (  
        benchmark : bool,   
        model_key : str,     
        moe_grid : list,
        x : jax.Array,
        reg : CoreRegistry,
        dimension : int,
        infB_reps : int,
        infB_qsize : int,
        inf_batch_size : int) -> CoreRegistry:
    '''
    \nCan be flagged to benchmark or not, either adding placeholder or actualy measurments to the core registry to maintain pipeline flow.
    \nRegisters the measured inference speed over N iterations.
    \nWe compute the median instead of average, since the average is more prone and unstable due to noise, which arise naturally from backrgound processes on the machine such as Thermal throtteling, Garbage collection, background tasks etc.
    '''
    print(f"\nInference Speed MoE:")

    # Measure inference for dense and sparse MoE
    cache = []

    # If we dont benchmark we fill core registry with filler
    if benchmark:
        speed = benchmark_inference_speed(
            x,
            moe_grid,
            moe_grid_forward_INF,
            inf_batch_size,
            infB_reps,
            infB_qsize,
            query_dim = x.shape[-1])
        cache.append([inf_batch_size, speed])
        
        print(f"Inf. Speed {inf_batch_size} batch size => {round(float(speed),4)}ms\n")
    else:
        cache.append([0, 0])
        print('Skipped Benchmarking')
    
    # Save results
    reg.add( model_key + core_keys['inf_speed_key'], cache)

    return reg