import jax
import jax.numpy as jnp

#------------------------------------------------------------------------------------

@jax.jit
def mlp_forward(p : list , X : jax.Array):
    for W in p[:-1]:
        # Bias Trick
        X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
        X = jax.nn.relu(X @ W)
    X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    return X @ p[-1]

@jax.jit
def mlp_forward_INF(p : list , x : jax.Array):
    return jax.nn.sigmoid(mlp_forward(p, x))

#------------------------------------------------------------------------------------

def mlp_error(x_batches : list, y : jax.Array, mlp : list):
    
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda X: mlp_forward_INF(mlp, X))(x_batched).flatten()

    ## Add tail
    x_tail = mlp_forward_INF(mlp, x_batches[-1])
    yp = jnp.concatenate((yp, x_tail.flatten()))

    # MSE 
    mse = jnp.mean((y - yp)**2)
    
    return mse, yp