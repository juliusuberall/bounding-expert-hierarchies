import jax
import jax.numpy as jnp
import optax

@jax.jit
def moe_forward_expert(p, x, idx, a=0.5, b=3.0):
    for e in p['experts'][:-1]:
        # We use the bias trick for all MoE implementation
        x = jnp.append(x, 1)
        x = jax.nn.tanh(jnp.dot(x, e[idx]))
    x = jnp.append(x, 1)
    return jax.nn.tanh(jnp.dot(x, p['experts'][-1][idx]) *a) *b

@jax.jit
def moe_forward_gate(p, x):
    for l in p['gate'][:-1]:
        x = jnp.append(x, 1)
        x = jax.nn.relu(jnp.dot(x, l))
    x = jnp.append(x, 1)
    return jax.nn.softmax(jnp.dot(x, p['gate'][-1]))

@jax.jit
def moe_forward_dense_full(p, x):
    activation = moe_forward_gate(p, x)
    x = jax.vmap(lambda idx: moe_forward_expert(p, x, idx))(jnp.arange(activation.shape[1]))
    return jnp.sum(x * jnp.expand_dims(activation, axis=-1))

@jax.jit
def moe_dense_KL_BCE_loss(p, x, y): 
    # Binary Cross-Entropy 
    yp = jax.vmap(lambda x: moe_forward_dense_full(p,x))(x)
    bce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(yp, y))

    # KL-divergence of the expert activation distribution against uniform distribution 
    activation = (jax.vmap(lambda x: moe_forward_gate(p, x))(x)) * jnp.expand_dims(y,axis=1)
    g = 1/jnp.sum(y) * jnp.sum(activation, axis=0) 
    kl_loss = jnp.sum(g * jnp.log(g / (1 / activation.shape[1])))

    return kl_loss + bce_loss

# FUNCTION: Validation loss over ALL or big chunk of data
# FUNCTION: Batch loss
# FUNCTION: Gradient visualizer