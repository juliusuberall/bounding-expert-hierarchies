import jax
import optax
import chex
import jax.numpy as jnp

from beh.core.moe import moe_forward_gate, moe_forward_dense
from beh.core.mlp import mlp_forward

#------------------------------------------------------------------------------------

@jax.jit
def sigmoid_binary_cross_entropy_focal_asymmetry(logits : jax.Array, labels : jax.Array, negative_class_weight : jax.Array):
    '''
    Neural bounding asymmetric BCE for achieving conservativness (https://dl.acm.org/doi/abs/10.1145/3641519.3657442).
    With additional focal term to cover outlier based on "Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)"
    Based on optax.sigmoid_binary_cross_entropy() implementation.
    '''
    chex.assert_type([logits], float)
    labels = labels.astype(logits.dtype)

    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)

    # Focal BCE to cover outlier based on "Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)"
    p = jax.nn.sigmoid(logits)
    p_t = p * labels + (1 - p) * (1 - labels)
    gamma = 2.0  # tune
    focal_weights = (1.0 - p_t) ** gamma

    # Asymmetric BCE diswaying ALL negatives as training progresses.
    bce = (-labels * log_p - (1.0 - labels) * log_not_p * negative_class_weight) * focal_weights

    return jnp.mean(bce)

#------------------------------------------------------------------------------------

@jax.jit
def sigmoid_binary_cross_entropy_asymmetry(logits : jax.Array, labels : jax.Array, negative_class_weight : jax.Array):
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
    yp = moe_forward_dense(p,x)
    bce_loss = sigmoid_binary_cross_entropy_focal_asymmetry(yp, y, negative_class_weight)

    # KL-divergence of the expert activation distribution against uniform distribution 
    activation = moe_forward_gate(p['gate'], x) * jnp.expand_dims(y,axis=1)
    nex = activation.shape[1]
    max_kl = jnp.log(nex)
    g = 1/jnp.sum(y) * jnp.sum(activation, axis=0) 
    kl_loss = jnp.sum(g * jnp.log(g / (1 / nex) + epsilon))

    # Gate Activation Entropy
    query_entropy = -jnp.sum(activation * jnp.log(activation + epsilon), axis=1)
    ae_loss = jnp.mean(query_entropy) * jnp.clip(-jnp.log(kl_loss / max_kl), 0, 1)

    return kl_loss + bce_loss + ae_loss

#------------------------------------------------------------------------------------

@jax.jit
def mlp_bce_loss(p : list, x : jax.Array, y : jax.Array, negative_class_weight : jax.Array):
    yp = mlp_forward(p,x).flatten() # Flatten to ensure correct shapes for BCE
    loss = sigmoid_binary_cross_entropy_asymmetry(yp, y, negative_class_weight)
    return loss