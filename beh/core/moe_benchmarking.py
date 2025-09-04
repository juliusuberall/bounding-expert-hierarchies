import jax
import jax.numpy as jnp

from beh.core.shared import *
from beh.core.moe import *
from beh.core.registry import *
from beh.core.benchmarking import *
from beh.core.moe_sparse import sparse_funcs_2048, sparse_funcs_200K

def register_accuracy(
        model_key : str,
        moe : dict,
        x_batches : list,
        y : jax.Array,
        reg : CoreRegistry,
        threshold : float,
    ):
    '''
    Computes and registers the accuracy with the provided training data batches.
    \nBased on those model outputs conservativness is computed and registered with false-negative,FN and false-positive,FP rates in %.
    '''
    # Dense MoE inference
    dense_mse, dense_yp, dense_yp_raw = moe_error(x_batches, y, moe, moe_forward_dense_INF)
    dense_fn, dense_fp = get_fn_fp_rate(dense_yp, y, threshold=threshold)

    # Sparse MoE inference
    sparse_mse, sparse_yp, sparse_yp_raw = moe_error(x_batches, y, moe, sparse_funcs_2048[model_key])
    sparse_fn, sparse_fp = get_fn_fp_rate(sparse_yp, y, threshold=threshold)

    # Save numerical results
    dkey = f'{model_key}_dense'
    skey = f'{model_key}_sparse'
    
    ## Dense
    reg.add(dkey + core_keys['accuracy_mse_key'],
            jnp.array(dense_mse))
    reg.add(dkey + core_keys['y_prediciton_key'],
        jnp.array(dense_yp))
    reg.add(dkey + core_keys['y_prediciton_RAW_key'],
        jnp.array(dense_yp_raw))
    reg.add(dkey + core_keys['fn_key'],
        jnp.array(dense_fn))
    reg.add(dkey + core_keys['fp_key'],
        jnp.array(dense_fp))

    
    ## Sparse
    reg.add(skey + core_keys['accuracy_mse_key'],
            jnp.array(sparse_mse))
    reg.add(skey + core_keys['y_prediciton_key'],
        jnp.array(sparse_yp))
    reg.add(skey + core_keys['y_prediciton_RAW_key'],
        jnp.array(sparse_yp_raw))
    reg.add(skey + core_keys['fn_key'],
        jnp.array(sparse_fn))
    reg.add(skey + core_keys['fp_key'],
        jnp.array(sparse_fp))

    print(f"\n      |    MSE   |    FN    |    FP   ")
    print(f"------|-------------------------------")
    print(f"Dense | {dense_mse:05f} | {dense_fn:05f} | {dense_fp:05f}")
    print(f"Sparse| {sparse_mse:05f} | {sparse_fn:05f} | {sparse_fp:05f}")

    return reg

def register_inference_speed (     
        model_key : str,     
        moe : dict,
        x : jax.Array,
        reg : CoreRegistry,
        dimension : int,
        infB_reps : int,
        infB_qsize : int,
        inf_batch_size : int) -> CoreRegistry:
    '''
    Registers the measured inference speed over N iterations.
    \nWe compute the minimum instead of average, since the average is more prone and unstable due to background noise, which arise naturally from backrgound processes on the machine such as Thermal throtteling, Garbage collection, background tasks etc.
    '''
    print(f"\nInference Speed MoE:")

    # Measure inference for dense and sparse MoE
    d_cache, s_cache = [], []
    dense_speed = benchmark_inference_speed(
        x,
        moe,
        moe_forward_dense_INF,
        inf_batch_size,
        infB_reps,
        infB_qsize,
        dimension)
    d_cache.append([inf_batch_size, dense_speed])

    sparse_speed = benchmark_inference_speed(
        x,
        moe,
        sparse_funcs_200K[model_key],
        inf_batch_size,
        infB_reps,
        infB_qsize,
        dimension)
    s_cache.append([inf_batch_size, sparse_speed])
    
    print(f"Dense Inf. Speed {inf_batch_size} batch size => {round(float(dense_speed),4)}ms")
    print(f"Sparse Inf. Speed {inf_batch_size} batch size => {round(float(sparse_speed),4)}ms\n")
    
    # Save results
    dkey = f'{model_key}_dense'
    skey = f'{model_key}_sparse'
    reg.add( dkey + core_keys['inf_speed_key'],
            d_cache)
    reg.add( skey + core_keys['inf_speed_key'],
            s_cache)

    return reg

def gating_confidence (     
        moe : dict,
        x_batches : list
    ):
    '''
    Computes MoE gating confidence metrics including:
    \n- Mean confidence 
    \n- Sorted gate activation
    \n- Topk
    '''
    # Compute mean for probability of top1 expert
    ## If gating probability not near 1.0 the MoE 
    ## relies on dense predicition
    x_batched = jnp.stack(x_batches[0:-1])
    gate_activation = jax.vmap(lambda x: moe_forward_gate_INF(moe, x))(x_batched)
    x_tail = moe_forward_gate_INF(moe, x_batches[-1])
    gate_activation = jnp.concatenate((gate_activation.reshape((-1,x_tail.shape[-1])), x_tail))

    # Extract crucial activation metrics
    _ , idx = jax.lax.top_k(gate_activation, 1)
    gate_sorted_activation = jnp.sort(gate_activation, axis=1)
    confidence = jnp.mean(jnp.max(gate_activation, axis=1).flatten())

    return confidence, gate_sorted_activation, idx

def register_gating_confidence ( 
        model_key : str,     
        moe : dict,
        x_batches : list,
        reg : CoreRegistry
    ):

    confidence, gate_sorted_activation, idx = gating_confidence(moe=moe,x_batches=x_batches)
    
    # Save numerical results
    reg.add( model_key + core_keys['gating_confidence_key'],
            confidence)
    reg.add( model_key + core_keys['gating_sorted_activation_key'],
            gate_sorted_activation)
    reg.add( model_key + core_keys['gate_top1_activation_key'],
            idx)

    print(f"\nGate Confidence: {round(float(confidence),4)}")
    return reg