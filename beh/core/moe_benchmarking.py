import jax
import jax.numpy as jnp
import optax

from beh.core.shared import *
from beh.core.moe import *
from beh.core.registry import *

def register_accuracy(
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
    dkey = 'moe' + '_dense_'
    skey = 'moe' + '_sparse_'
    
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

def get_fn_fp_rate(yp : jax.Array, y : jax.Array, threshold : float = 0.1):
    '''
    Measure conservativness of predictions and labels with False-Negative and False-Positive rate.
    '''
    fp_rate = jnp.sum((yp > threshold) * ( y == 0)) / y.size
    fn_rate = jnp.sum((yp < threshold) * ( y > 0)) / y.size
    return fn_rate, fp_rate

def register_gating_confidence (       
        moe : dict,
        x_batches : list,
        reg : CoreRegistry
    ):
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

    
    # Save numerical results
    reg.add( 'moe' + core_keys['gating_confidence_key'],
            confidence)
    reg.add( 'moe' + core_keys['gating_sorted_activation_key'],
            gate_sorted_activation)
    reg.add( 'moe' + core_keys['gate_top1_activation_key'],
            idx)

    print(f"\nMoE Confidence: {round(float(confidence),4)}")
    return reg

def register_all_expert_boundaries(
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
    reg.add( 'moe' + core_keys['expert_boundary_key'],
            e_decBoundaries)
    
    return reg