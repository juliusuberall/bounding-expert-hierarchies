import jax
import jax.numpy as jnp
import numpy as np

from beh.adapter.shared import *
from beh.core.shared import remap
from beh.core.moe import expert_forward_sparse
from beh.registry import *
from beh.config_parser import *

args, configs = parse_train()

#------------------------------------------------------------------------------------

@jax.jit
def moe_grid_forward_INF(experts : list, x : jax.Array):
    idx = moe_grid_select(x)
    return jax.nn.sigmoid(expert_forward_sparse(experts, x, idx))

#------------------------------------------------------------------------------------

@jax.jit
def moe_grid_select(x : jax.Array):
    """Map queries to networks by computing networks index in grid for any query dimension.
    \nFor 4D+ dimensions this may seem non-trivial but can be generated. 
    \nIt essentially checks a grid and assigns based on this grid cell index."""
    # Get number of grid cells per dimension
    grid_dim = configs['moeGrid']['grid_dim']

    # Map training data (normalised in range -1 to 1) to grid cells
    dimension = x.shape[-1]
    grid_mapping = jnp.floor(remap(x, -1, 1, 0, grid_dim - 1e-4))

    # Convert mapping to grid cell index used for selecting NN
    idx = jnp.zeros(grid_mapping.shape[:-1])
    for i in range(dimension):
        idx += grid_mapping[..., i] * grid_dim ** i
    
    return idx.astype(jnp.int32)

#------------------------------------------------------------------------------------

def batch_query_moe_grid(x_batches : list, experts : list, remap_flag : bool = True):
    '''List of batched queries that will be passed through the model at once. Be aware of device OOM.'''
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    
    yp = jax.vmap(lambda x: moe_grid_forward_INF(experts, x))(x_batched).flatten()
    yp_tail = moe_grid_forward_INF(experts, x_batches[-1]).flatten()
    yp = jnp.concatenate((yp, yp_tail), axis=0)

    idx = moe_grid_select(x_batched)
    idx_tail = moe_grid_select(x_batches[-1])
    idx = jnp.concatenate((idx.flatten(), idx_tail), axis=0)

    # Remap
    yp_raw = yp.copy()
    if remap_flag:
        yp = remap(
            yp, 
            jnp.min(yp),
            jnp.max(yp),
            0,
            1,)

    return yp, idx, yp_raw

#------------------------------------------------------------------------------------

def batch_query_moe_grid_OOM(x_batches : list, experts : list, at_once : int):
    '''List of batched queries that will be passed through the model by looping over subsets of batches to avoid OOM on device when passing all batches at once.
    \nPassing all batches at once can be done with batch_query_moe().'''

    # Forward batch subsets WITHOUT remapping the subsets model output already to 0.0 - 1.0
    yp_all = []
    for i in range(0, len(x_batches), at_once):
        yp, _ , _ = batch_query_moe_grid(x_batches[i:i+at_once], experts, remap_flag=False)
        yp_all.append(np.array(yp)) # unload from GPU
    yp_all = np.concatenate(yp_all, axis=0)

    # Remap after all batches were processed to avoid uneven remapping
    yp_all = remap(
        yp_all, 
        np.min(yp_all),
        np.max(yp_all),
        0,
        1,)
    
    return yp_all