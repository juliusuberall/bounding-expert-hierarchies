import jax
import jax.numpy as jnp
import optax

#------------------------------------------------------------------------------------

@jax.jit
def mlp_forward(p : list , x : jax.Array):
    for w in p[:-1]:
        # Bias Trick
        x = jnp.append(x, 1)
        x = jax.nn.relu(jnp.dot(x, w))
    x = jnp.append(x, 1)
    return jnp.dot(x, p[-1])

@jax.jit
def mlp_forward_INF(p : list , x : jax.Array):
    return jax.nn.sigmoid(mlp_forward(p, x))

#------------------------------------------------------------------------------------

def mlp_error(x_batches : list, y : jax.Array, mlp : list):
    
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda batch: 
                jax.vmap(lambda x: mlp_forward_INF(mlp, x))(batch)
                )(x_batched).flatten()

    ## Add tail
    x_tail = jax.vmap(lambda x: mlp_forward_INF(mlp,x))(x_batches[-1])
    yp = jnp.concatenate((yp, x_tail.flatten()))

    # MSE 
    mse = jnp.mean((y - yp)**2)
    
    return mse, yp