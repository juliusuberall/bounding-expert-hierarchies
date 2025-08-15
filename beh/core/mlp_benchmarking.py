import jax
import jax.numpy as jnp

from beh.core.shared import *
from beh.core.mlp import *
from beh.core.registry import *
from beh.core.benchmarking import *

def register_accuracy(
        model_key : str,
        mlp : list,
        x_batches : list,
        y : jax.Array,
        reg : CoreRegistry,
        threshold : float,
    ):
    '''
    Computes and registers the accuracy with the provided training data batches.
    \nBased on those model outputs conservativness is computed and registered with false-negative,FN and false-positive,FP rates in %.
    '''
    
    # MLP inference
    mse, yp = mlp_error(x_batches, y, mlp)
    fn, fp = get_fn_fp_rate(yp, y, threshold=threshold)

    # Save numerical results
    reg.add(model_key + core_keys['accuracy_mse_key'],
            jnp.array(mse))
    reg.add(model_key + core_keys['y_prediciton_key'],
        jnp.array(yp))
    reg.add(model_key + core_keys['fn_key'],
        jnp.array(fn))
    reg.add(model_key + core_keys['fp_key'],
        jnp.array(fp))


    print(f"\nMSE: {round(float(mse),4)}")
    print(f"FN: {float(fn)}")
    print(f"FP: {float(fp)}")

    return reg

def register_inference_speed (     
        model_key : str,     
        mlp : list,
        x : jax.Array,
        reg : CoreRegistry,
        configs : dict,
        dimension : int,
        infB_reps : int,
        infB_qsize : int ) -> CoreRegistry:
    '''
    Registers the measured minimum inference speed over N iterations.
    \nWe compute the minimum instead of average, since the average is more prone and unstable due to background noise, which arise naturally from backrgound processes on the machine such as Thermal throtteling, Garbage collection, background tasks etc.
    '''
    print(f"\nMin. Inference Speed MLP:")

    # Measure inference for dense and sparse MoE
    speed = min_inference_speed(
        x,
        mlp,
        mlp_forward_INF,
        infB_reps,
        infB_qsize,
        configs,
        dimension)
    
    # Save numerical results
    reg.add( model_key + core_keys['inf_speed_key'],
            speed)

    print(f"{round(float(speed),4)}ms")

    return reg