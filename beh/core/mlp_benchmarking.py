import jax
import jax.numpy as jnp
import numpy as np

from beh.core.shared import *
from beh.core.mlp import *
from beh.core.registry import *
from beh.core.benchmarking import *

def register_accuracy(
        model_key : str,
        mlp : list,
        x_batches : list,
        x_aa : list,
        y : jax.Array,
        reg : CoreRegistry,
        threshold : float,
    ):
    '''
    Computes and registers the accuracy with the provided training data batches.
    \nBased on those model outputs conservativness is computed and registered with false-negative,FN and false-positive,FP rates in %.
    '''
    
    # MLP inference
    yp = batch_query_mlp(x_batches, mlp)
    fn, fp = get_fn_fp_rate(yp, y, threshold=threshold)

    ## 2D - Query in higher resolution for anti-aliased binary classification plot
    if x_aa != None : 
        yp_aa = batch_query_mlp(x_aa, mlp)

    # Save numerical results
    reg.add(model_key + core_keys['y_prediciton_key'], yp)
    reg.add(model_key + core_keys['fn_key'], fn)
    reg.add(model_key + core_keys['fp_key'], fp)
    reg.add(model_key + core_keys['aa_y_prediciton_key'], yp_aa)

    print(f"FN: {float(fn)}")
    print(f"FP: {float(fp)}")

    return reg

def register_inference_speed (     
        model_key : str,     
        mlp : list,
        x : jax.Array,
        reg : CoreRegistry,
        dimension : int,
        infB_reps : int,
        infB_qsize : int,
        inf_batch_size : int) -> CoreRegistry:
    '''
    Registers the measured minimum inference speed over N iterations.
    \nWe compute the minimum instead of average, since the average is more prone and unstable due to background noise, which arise naturally from backrgound processes on the machine such as Thermal throtteling, Garbage collection, background tasks etc.
    '''
    print(f"\nInference Speed MLP:")

    # Measure inference speed for each batchsize
    cache = []
    speed = benchmark_inference_speed(
        x,
        mlp,
        mlp_forward_INF,
        inf_batch_size,
        infB_reps,
        infB_qsize,
        dimension)
    cache.append([inf_batch_size, speed])
    print(f"\nInf. Speed {inf_batch_size} batch size => {round(float(speed),4)}ms")
    
    # Save all speeds paired with their batchsize
    reg.add( model_key + core_keys['inf_speed_key'], cache)

    return reg