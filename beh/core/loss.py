import jax
import optax
import chex
import jax.numpy as jnp

from beh.core.moe import moe_forward_gate, moe_forward_dense
from beh.core.mlp import mlp_forward

#------------------------------------------------------------------------------------

@jax.jit
def self_balancing_sigmoid_binary_cross_entropy(logits : jax.Array, labels : jax.Array, negative_class_weight : jax.Array):
    '''
    Neural bounding asymmetric BCE for achieving conservativness (https://dl.acm.org/doi/abs/10.1145/3641519.3657442).
    Based on optax.sigmoid_binary_cross_entropy() implementation.
    '''
    chex.assert_type([logits], float)
    labels = labels.astype(logits.dtype)

    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)

    # Asymmetric BCE diswaying ALL negatives as training progresses.
    bce = -labels * log_p - (1.0 - labels) * log_not_p * negative_class_weight

    return jnp.mean(bce)

#------------------------------------------------------------------------------------

@jax.jit
def moe_train_loss(p : dict, x : jax.Array, y : jax.Array, negative_class_weight : jax.Array):
    '''
    Deafult MoE loss. Combines
    \n- BCE -> Conservativness and bounding
    \n- KL gating -> Uniform expert usage
    \n- Gating Entropy -> Sparse expert usage while training dense
    '''
    epsilon = 1e-8

    # Binary Cross-Entropy 
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = self_balancing_sigmoid_binary_cross_entropy(yp, y, negative_class_weight)

    # KL-divergence of the expert activation distribution against uniform distribution 
    activation = (jax.vmap(lambda x: moe_forward_gate(p, x))(x)) * jnp.expand_dims(y,axis=1)
    nex = activation.shape[1]
    max_kl = jnp.log(nex)
    g = 1/jnp.sum(y) * jnp.sum(activation, axis=0) 
    kl_loss = jnp.sum(g * jnp.log(g / (1 / nex) + epsilon)) / max_kl

    # Gate Activation Entropy
    query_entropy = -jnp.sum(activation * jnp.log(activation + epsilon), axis=1)
    ae_loss = jnp.mean(query_entropy) * jnp.clip(-jnp.log(kl_loss), 0, 1)

    return kl_loss + bce_loss + ae_loss

#------------------------------------------------------------------------------------

@jax.jit
def moe_train_loss_1_expert(p : dict, x : jax.Array, y : jax.Array, negative_class_weight : jax.Array):
    '''Intended for benchmark MoE with only 1 expert. Will only use BCE, since gate needs no specific loss.'''
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = self_balancing_sigmoid_binary_cross_entropy(yp, y, negative_class_weight)
    return bce_loss

#------------------------------------------------------------------------------------

@jax.jit
def mlp_bce_loss(p : list, x : jax.Array, y : jax.Array, negative_class_weight : jax.Array):
    yp = jax.vmap(lambda x: mlp_forward(p,x))(x).flatten() # Flatten to ensure correct shapes for BCE
    loss = self_balancing_sigmoid_binary_cross_entropy(yp, y, negative_class_weight)
    return loss