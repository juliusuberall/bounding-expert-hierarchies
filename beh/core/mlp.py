import jax
import jax.numpy as jnp
import optax

#------------------------------------------------------------------------------------

@jax.jit
def mlp_fastforward(p : list , x : jax.Array):
    for W, b in p[:-1]:
        x = jax.nn.relu(jnp.dot(x, W) + b)
    final_w, final_b = p[-1]
    return jnp.dot(x, final_w) + final_b

@jax.jit
def mlp_fastforward_INF(p : list , x : jax.Array):
    return jax.nn.sigmoid(mlp_fastforward(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def mlp_forward(p : list , x : jax. Array):
    return jax.vmap(lambda x: mlp_fastforward(p, x))(x).flatten()

@jax.jit
def mlp_forward_INF(p : list , x : jax.Array):
    return jax.nn.sigmoid(mlp_forward(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def mlp_bce_loss(p : list, x : jax.Array, y : jax.Array):
    preds = mlp_forward(p, x)
    loss = optax.sigmoid_binary_cross_entropy(preds, y)
    return jnp.mean(loss)

#------------------------------------------------------------------------------------

def mlp_error(x_batches : list, y : jax.Array, p : list):
    
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda x: mlp_forward_INF(p, x))(x_batched).flatten()

    ## Add tail
    x_tail = mlp_forward_INF(p, x_batches[-1])
    yp = jnp.concatenate((yp, x_tail.flatten()))

    # MSE 
    mse = jnp.mean((y - yp)**2)
    return mse, yp