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
        grid_mapping = jnp.floor(remap(x, -1, 1, 0, grid_dim - 1e-8))
        # Convert mapping to grid cell index used for selecting NN
        idx = (grid_mapping[...,0] * grid_dim + grid_mapping[...,1] * (grid_dim - 1)).astype(jnp.int32)
        return idx

#------------------------------------------------------------------------------------

def batch_query_moe_grid(x_batches : list, experts : list, func, dimension: int, grid_dim : int, remap_flag : bool = True):
    '''List of batched queries that will be passed through the model at once. Be aware of device OOM.'''
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])

    # Originally we used vmap here but this caused for large models always OOM on a T4, 
    # which is why we swapped to a on devcice sequential compute, forward passing the batch
    # through each expert.
    def run_in_batches(func, experts, x_batches):
        def step(carry, xb):
            idx = moe_grid_select(xb, dimension, grid_dim)
            yp = func(experts, xb, idx).flatten()
            return carry, (yp, idx)

        _, out = jax.lax.scan(step, None, x_batched)
        yp, idx = out[0], out[1]
        return jnp.concatenate(yp, axis=0), jnp.concatenate(idx, axis=0)

    yp, idx = run_in_batches(func, experts, x_batched)
    idx_tail = moe_grid_select(x_batches[-1], dimension, grid_dim)
    yp_tail = func(experts, x_batches[-1], idx_tail).flatten()
    yp = jnp.concatenate((yp, yp_tail), axis=0)
    idx = jnp.concatenate((idx, idx_tail), axis=0)

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