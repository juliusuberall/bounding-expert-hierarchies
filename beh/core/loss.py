import jax
import optax
import chex
import jax.numpy as jnp

from beh.core.moe import moe_forward_gate, moe_forward_dense
from beh.core.mlp import mlp_forward

#------------------------------------------------------------------------------------

@jax.jit
def self_balancing_sigmoid_binary_cross_entropy(logits : jax.Array, labels : jax.Array, self_balance : jax.Array):
    '''
    Based on optax.sigmoid_binary_cross_entropy() implementation.
    \nExtended to ensure conservativness through BCE that can self balance itself. 
    \nDepending on the positive label coverage it either enforces the fitting of positives or regulates itself to a classic BCE with 1:1 weigthing.
    \nIdeally the self-balance increases linearly throughout training to ensure a smooth handover.
    \nSelf balancing factor must be in range 0.0 - 1.0
    '''
    chex.assert_type([logits], float)
    labels = labels.astype(logits.dtype)

    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)

    # Positive sample coverage used for self balancing
    a = jnp.clip(jnp.sum(log_p * labels), -5, 0)
    n_weight = jnp.exp(a)
    n_weight += (1 - n_weight) * (1 - self_balance)

    return -labels * log_p - (1.0 - labels) * log_not_p * n_weight

#------------------------------------------------------------------------------------

@jax.jit
def moe_train_loss(p : dict, x : jax.Array, y : jax.Array, self_balance : jax.Array):
    epsilon = 1e-8

    # Binary Cross-Entropy 
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = jnp.mean(self_balancing_sigmoid_binary_cross_entropy(yp, y, self_balance))

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
def mlp_bce_loss(p : list, x : jax.Array, y : jax.Array, self_balance : jax.Array):
    yp = jax.vmap(lambda x: mlp_forward(p,x))(x).flatten() # Flatten to ensure correct shapes for BCE
    loss = jnp.mean(self_balancing_sigmoid_binary_cross_entropy(yp, y, self_balance))
    return loss