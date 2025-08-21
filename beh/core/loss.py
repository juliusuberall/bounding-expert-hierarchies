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

    # Computing individual sample factors for weighted average of BCE.
    ## Clip to ensure when boundary gets tight and confident we 
    ## avoid loss explosions for some False-Negatives.
    ## If confident False Positve we just remove it fully from 
    ## loss to have no impact as we dont care about False-Positives
    ## and diminish their BCE contribution anyway.
    exp_fn = jnp.clip(jnp.exp(-labels * log_p), 1, 10) 
    exp_fp = jnp.exp((1.0 - labels) * log_not_p)
    exp_bce = exp_fn + exp_fp

    # Asymmetric BCE diswaying ALL negatives linearly as training progresses.
    # We avoid factoring out all negatives to avoid major optimization shift
    # and resulting unstable training.
    bce = -labels * log_p - (1.0 - labels) * log_not_p * (1.01 - self_balance)
    
    # Increasing the FN focus and decreasing FP focus individually based on model 
    # confidence for FN and FP linearly as training progresses.
    # Creates better results as without doing this in parallel
    bce =  bce * (exp_bce**self_balance)

    return jnp.mean(bce)

#------------------------------------------------------------------------------------

@jax.jit
def moe_train_loss(p : dict, x : jax.Array, y : jax.Array, self_balance : jax.Array):
    '''
    Deafult MoE loss. Combines
    \n- BCE -> Conservativness and bounding
    \n- KL gating -> Uniform expert usage
    \n- Gating Entropy -> Sparse expert usage while training dense
    '''
    epsilon = 1e-8

    # Binary Cross-Entropy 
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = self_balancing_sigmoid_binary_cross_entropy(yp, y, self_balance)

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
def moe_train_loss_1_expert(p : dict, x : jax.Array, y : jax.Array, self_balance : jax.Array):
    '''Intended for benchmark MoE with only 1 expert. Will only use BCE, since gate needs no specific loss.'''
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = self_balancing_sigmoid_binary_cross_entropy(yp, y, self_balance)
    return bce_loss

#------------------------------------------------------------------------------------

@jax.jit
def mlp_bce_loss(p : list, x : jax.Array, y : jax.Array, self_balance : jax.Array):
    yp = jax.vmap(lambda x: mlp_forward(p,x))(x).flatten() # Flatten to ensure correct shapes for BCE
    loss = self_balancing_sigmoid_binary_cross_entropy(yp, y, self_balance)
    return loss