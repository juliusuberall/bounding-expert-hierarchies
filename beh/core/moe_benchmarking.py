import jax
import jax.numpy as jnp

from beh.core.shared import *
from beh.core.moe import *
from beh.core.registry import *
from beh.core.benchmarking import *

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

    # Sparse MoE inference
    sparse_mse, sparse_yp, sparse_yp_raw = moe_error(x_batches, y, moe, moe_forward_sparse_INF)
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
    
    ## Sparse
    reg.add(skey + core_keys['accuracy_mse_key'],
            jnp.array(sparse_mse))
    reg.add(skey + core_keys['y_prediciton_key'],
        jnp.array(sparse_yp))
    reg.add(skey + core_keys['y_prediciton_RAW_key'],
        jnp.array(sparse_yp_raw))
    reg.add(skey + core_keys['sparse_fn_key'],
        jnp.array(sparse_fn))
    reg.add(skey + core_keys['sparse_fp_key'],
        jnp.array(sparse_fp))

    print(f"\nDense MSE: {round(float(dense_mse),4)}")
    print(f"Sparse MSE: {round(float(sparse_mse),4)}")
    print(f"Sparse FN: {float(sparse_fn)}")
    print(f"Sparse FP: {float(sparse_fp)}")

    return reg

def register_inference_speed (     
        model_key : str,     
        moe : dict,
        x_batches : list,
        reg : CoreRegistry,
        configs : dict,
        dimension : int ) -> CoreRegistry:
    '''
    Registers the measured minimum inference speed over N iterations.
    \nWe compute the minimum instead of average, since the average is more prone and unstable due to background noise, which arise naturally from backrgound processes on the machine such as Thermal throtteling, Garbage collection, background tasks etc.
    '''
    print(f"\nMin. Inference Speed MoE:")

    # Measure inference for dense and sparse MoE
    iterations = 100
    batch_repitions = 10
    
    dense_speed = min_inference_speed(
        x_batches,
        moe,
        moe_forward_dense_INF,
        iterations,
        batch_repitions,
        configs,
        dimension)
    
    sparse_speed = min_inference_speed(
        x_batches,
        moe,
        moe_forward_sparse_INF,
        iterations,
        batch_repitions,
        configs,
        dimension)
    
    # Save numerical results
    reg.add( model_key + core_keys['dense_inf_speed_key'],
            dense_speed)
    reg.add( model_key + core_keys['sparse_inf_speed_key'],
            sparse_speed)

    print(f"Dense: {round(float(dense_speed),4)}µs")
    print(f"Sparse: {round(float(sparse_speed),4)}µs")

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
    gate_activation = jax.vmap(lambda x: 
                    jax.vmap(lambda x: moe_forward_gate_INF(moe, x))(x)
                    )(x_batched)
    x_tail = jax.vmap(lambda x: moe_forward_gate_INF(moe,x))(x_batches[-1])
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

def register_all_expert_boundaries(
        model_key : str,
        moe : dict,
        x_batches : list,
        reg : CoreRegistry
    ):

    # Retrieve number of experts
    nex = moe['experts'][0].shape[0]

    # Get prediction boundaries for full training signal from each expert
    e_decBoundaries = []
    x_batched = jnp.stack(x_batches[0:-1])
    for expert_i in range(nex):
        iyp = jax.vmap(lambda x: 
                        jax.vmap(lambda x: moe_forward_expert_INF(moe, x, expert_i))(x)
                        )(x_batched).flatten()
        ## Add tail
        x_tail = jax.vmap(lambda x: moe_forward_expert_INF(moe, x, expert_i))(x_batches[-1])
        e_decBoundaries.append(jnp.concatenate((iyp, x_tail.flatten())))
    
    # Save numerical results
    reg.add( model_key + core_keys['expert_boundary_key'],
            e_decBoundaries)
    
    return reg