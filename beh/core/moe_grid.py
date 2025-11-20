import jax
import jax.numpy as jnp
import numpy as np

from datetime import datetime

from beh.adapter.shared import *
from beh.core.shared import remap
from beh.registry import *
from beh.config_parser import *
from beh.core.moe import moe_forward_expert, moe_forward_expert_INF

@jax.jit
def moe_grid_forward(experts : list, x : jax.Array, idx : jax.Array):
    # Select parameters on per query basis
    for e in experts[:-1]:
        # Bias Trick
        x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
        x = jax.nn.tanh(jnp.einsum('bi,bij->bj', x, e[idx]))
    x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    # We found that these additional factors improved the accurarcy as well as the "double" activation with tanh() and sigmoid() in the final layer.
    # The sigmoid is not defined here, because OPTAX does this for numerical stability in their BCE implementation. Therefore the regualr model output
    # needs to be additionally transformed with sigmoid()
    return jax.nn.tanh(jnp.einsum('bi,bij->bj', x, experts[-1][idx])* 0.5) * 3.0 

#------------------------------------------------------------------------------------

def moe_grid_select(x : jax.Array, dimension: int, grid_dim : int):
    """Map the queries to the networks. This simply checks the grid and assigns based on this grid."""
    if dimension == 2:
        # Map training data (normalised in range -1 to 1) to grid cells
        grid_mapping = jnp.floor(remap(x, -1, 1, 0, grid_dim - 1e-4))
        # Convert mapping to grid cell index used for selecting NN
        idx = (grid_mapping[...,0] * grid_dim + grid_mapping[...,1]).astype(jnp.int32)
        return idx

#------------------------------------------------------------------------------------

def batch_query_moe_grid(x_batches : list, experts : list, dimension: int, grid_dim : int, remap_flag : bool = True):
    '''List of batched queries that will be passed through the model at once. Be aware of device OOM.'''
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    
    idx = moe_grid_select(x_batched, dimension, grid_dim)
    yp = jax.vmap(lambda x, i: moe_grid_forward_INF(experts, x, i))(x_batched, idx).flatten()

    idx_tail = moe_grid_select(x_batches[-1], dimension, grid_dim)
    yp_tail = moe_grid_forward_INF(experts, x_batches[-1], idx_tail).flatten()
    yp = jnp.concatenate((yp, yp_tail), axis=0)
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

def batch_query_moe_grid_OOM(x_batches : list, experts : list, at_once : int,  dimension: int, grid_dim : int):
    '''List of batched queries that will be passed through the model by looping over subsets of batches to avoid OOM on device when passing all batches at once.
    \nPassing all batches at once can be done with batch_query_moe().'''

    # Forward batch subsets WITHOUT remapping the subsets model output already to 0.0 - 1.0
    yp_all = []
    for i in range(0, len(x_batches), at_once):
        yp, _ , _ = batch_query_moe_grid(x_batches[i:i+at_once], experts, dimension, grid_dim, remap_flag=False)
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