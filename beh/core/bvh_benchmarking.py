import jax
import jax.numpy as jnp
import numpy as np

from beh.core.shared import *
from beh.core.bvh import *
from beh.core.registry import *
from beh.core.benchmarking import *

def register_accuracy(
        model_key : str,
        bvh : jax.Array,
        x : jax.Array,
        x_aa : list,
        y : jax.Array,
        reg : CoreRegistry,
        threshold : float,
    ):
    '''
    Computes and registers the accuracy with the provided training data.
    \nBased on those model outputs conservativness is computed and registered with false-negative,FN and false-positive,FP rates in % of max FN or FP that could be reached.
    '''
    
    # Inference
    yp, idx = batch_query_bvh(bvh, x)
    fn, fp = get_fn_fp_rate(yp, y, threshold)

    ## 2D - Query in higher resolution for anti-aliased binary classification plot
    # if x_aa != None : 
    #     yp_aa = batch_query_bvh(bvh, jnp.array(x_aa).reshape((-1, x.shape[-1])))
    #     reg.add(model_key + core_keys['aa_y_prediciton_key'], yp_aa)

    # Save numerical results
    reg.add(model_key + core_keys['y_prediciton_key'], yp)
    reg.add(model_key + core_keys['gate_top1_activation_key'], idx)
    reg.add(model_key + core_keys['fn_key'], fn)
    reg.add(model_key + core_keys['fp_key'], fp)

    print(f"FN: {float(fn)}")
    print(f"FP: {float(fp)}")

    return reg

def register_inference_speed (  
        benchmark : bool,   
        model_key : str,     
        bvh : jax.Array,
        x : jax.Array,
        reg : CoreRegistry,
        dimension : int,
        infB_reps : int,
        infB_qsize : int,
        infB_batch_size : int) -> CoreRegistry:
    '''
    \nCan be flagged to benchmark or not, either adding placeholder or actualy measurments to the core registry to maintain pipeline flow.
    \nRegisters the measured median inference speed over N iterations.
    \nWe compute the median instead of average, since the average is more prone and unstable due to noise, which arise naturally from backrgound processes on the machine such as Thermal throtteling, Garbage collection, background tasks etc.
    '''
    print(f"\nInference Speed MLP:")

    # Measure inference speed for each batchsize
    cache = []
    if benchmark:
        speed = benchmark_inference_speed(
            x,
            bvh,
            query_bvh,
            infB_batch_size,
            infB_reps,
            infB_qsize,
            query_dim = x.shape[-1])
        cache.append([infB_batch_size, speed])
        print(f"\nInf. Speed {infB_batch_size} batch size => {round(float(speed),4)}ms")
    else:
        cache.append([0, 0])
        print('Skipped Benchmarking')

    # Save all speeds paired with their batchsize
    reg.add( model_key + core_keys['inf_speed_key'], cache)

    return reg