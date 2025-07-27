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
        reg : CoreRegistry
    ):

    # Dense MoE inference
    dense_mse, dense_yp, dense_yp_raw = moe_loss(x_batches, y, moe, moe_forward_dense_INF)

    # Sparse MoE inference
    sparse_mse, sparse_yp, sparse_yp_raw = moe_loss(x_batches, y, moe, moe_forward_sparse_INF)

    # Save numerical results
    dkey = 'moe' + '_dense_'
    skey = 'moe' + '_sparse_'

    reg.add(dkey + core_keys['accuracy_mse_key'],
            jnp.array(dense_mse))
    reg.add(dkey + core_keys['y_prediciton_key'],
        jnp.array(dense_yp))
    reg.add(dkey + core_keys['y_prediciton_RAW_key'],
        jnp.array(dense_yp_raw))
    
    reg.add(skey + core_keys['accuracy_mse_key'],
            jnp.array(sparse_mse))
    reg.add(skey + core_keys['y_prediciton_key'],
        jnp.array(sparse_yp))
    reg.add(skey + core_keys['y_prediciton_RAW_key'],
        jnp.array(sparse_yp_raw))

    print(f"\nDense Prediction MSE: {round(float(dense_mse),4)}")
    print(f"Sparse Prediction MSE: {round(float(sparse_mse),4)}")

    return reg

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
    gate_activation = jnp.concatenate((gate_activation, jnp.expand_dims(x_tail, axis=0)))

    # Extract crucial activation metrics
    _ , idx = jax.lax.top_k(gate_activation, 1)
    gate_sorted_activation = jnp.sort(gate_activation, axis=2)
    confidence = jnp.mean(jnp.max(gate_activation, axis=2).flatten())

    
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