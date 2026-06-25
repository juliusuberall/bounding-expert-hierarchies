import jax
import jax.numpy as jnp
import numpy as np

from beh.config_parser import parse_train
from beh.core.shared import positional_encoding

_ , configs = parse_train()

#------------------------------------------------------------------------------------

@jax.jit
def mlp_forward(p : list , X : jax.Array):
    X = positional_encoding(X, configs['general']['pe_num_freq'])
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

def batch_query_mlp(x_batches : list, mlp : list):
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda X: mlp_forward_INF(mlp, X))(x_batched).flatten()
    ## Add tail
    x_tail = mlp_forward_INF(mlp, x_batches[-1])
    yp = jnp.concatenate((yp, x_tail.flatten()))
    return yp

#------------------------------------------------------------------------------------

def batch_query_mlp_OOM(x_batches : list, mlp : list, at_once : int):
    '''List of batched queries that will be passed through the model by looping over subsets of batches to avoid OOM on device when passing all batches at once.
    \nPassing all batches at once can be done with batch_query_mlp().'''

    yp_all = []
    for i in range(0, len(x_batches), at_once):
        yp = batch_query_mlp(x_batches[i:i+at_once], mlp)
        yp_all.append(np.array(yp)) # unload from GPU
    yp_all = np.concatenate(yp_all, axis=0)
    
    return yp_all