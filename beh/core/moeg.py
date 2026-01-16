import jax
import jax.numpy as jnp
import numpy as np

from beh.core.shared import remap
from beh.core.moe import expert_forward_sparse
from beh.core.registry import gating_table
from beh.config_parser import parse_train

args , _ = parse_train()

#------------------------------------------------------------------------------------

@jax.jit
def moeg_forward_INF(experts : list, x : jax.Array):
    idx = moeg_select(x, experts)
    return jax.nn.sigmoid(expert_forward_sparse(experts, x, idx))

#------------------------------------------------------------------------------------

@jax.jit
def moeg_select(x : jax.Array, experts : list):
    """Map queries to networks by computing networks index in grid for any query dimension.
    \nFor 4D+ dimensions this may seem non-trivial but can be generated. 
    \nIt essentially checks a grid and assigns based on this grid cell index."""
    
    # Grid dimension
    ## We assign queries of points and rays (point + direction) always based on their point location.
    ## We assume the first n-D coordinates to always represent the point location.
    dimension = gating_table[args.query][x.shape[-1]]

    # Number of grid cells per dimension in hypercube
    grid_dim = experts[0].shape[0] ** (1/dimension)

    # Map training data (normalised in range -1 to 1) to grid cells
    grid_mapping = jnp.floor(remap(x[...,:dimension], -1, 1, 0, grid_dim - 1e-4))

    # Convert mapping to grid cell index used for selecting NN
    idx = jnp.zeros(grid_mapping.shape[:-1])
    for i in range(dimension):
        idx += grid_mapping[..., i] * grid_dim ** i
    
    return idx.astype(jnp.int32)

#------------------------------------------------------------------------------------

def batch_query_moeg(x_batches : list, experts : list, remap_flag : bool = True):
    '''List of batched queries that will be passed through the model at once. Be aware of device OOM.'''
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    
    yp = jax.vmap(lambda x: moeg_forward_INF(experts, x))(x_batched).flatten()
    yp_tail = moeg_forward_INF(experts, x_batches[-1]).flatten()
    yp = jnp.concatenate((yp, yp_tail), axis=0)

    idx = moeg_select(x_batched, experts)
    idx_tail = moeg_select(x_batches[-1], experts)
    idx = jnp.concatenate((idx.flatten(), idx_tail), axis=0)
    yp_raw = yp

    return yp, idx, yp_raw

#------------------------------------------------------------------------------------

def batch_query_moeg_OOM(x_batches : list, experts : list, at_once : int):
    '''List of batched queries that will be passed through the model by looping over subsets of batches to avoid OOM on device when passing all batches at once.
    \nPassing all batches at once can be done with batch_query_moe().'''

    yp_all, idx_all = [], []
    for i in range(0, len(x_batches), at_once):
        yp, idx , _ = batch_query_moeg(x_batches[i:i+at_once], experts, remap_flag=False)
        # Unload from GPU
        yp_all.append(np.array(yp))
        idx_all.append(np.array(idx))

    yp_all = np.concatenate(yp_all, axis=0)
    idx_all = np.concatenate(idx_all, axis=0)
    
    return yp_all, idx_all