import jax
import jax.numpy as jnp
import numpy as np

from datetime import datetime

from beh.adapter.shared import *
from beh.core.shared import *
from beh.registry import *
from beh.config_parser import *

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_expert(p : list, x : jax.Array):
    for e in p[:-1]:
        # Bias Trick
        x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
        x = jax.nn.tanh(x @ e)
    x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    return jax.nn.tanh((x @ p[-1]) * 0.5) * 3.0 # 3D jax.nn.tanh((x @ p[-1]) * 0.01) * 10.0

@jax.jit
def moe_forward_expert_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_expert(p['experts'], x))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_gate(p : list, x : jax.Array):
    for l in p[:-1]:
        x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
        x = jax.nn.relu(x @ l)
    x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    return jax.nn.softmax(x @ p[-1])

@jax.jit
def moe_forward_gate_INF(p : dict, x : jax.Array):
    return moe_forward_gate(p['gate'], x)

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_dense(p : dict, x : jax.Array):
    activation = moe_forward_gate(p['gate'], x)
    y = jax.vmap(lambda p: moe_forward_expert(p, x), out_axes=1)(p['experts']).squeeze()
    return jnp.sum(y * activation, axis=-1)

@jax.jit
def moe_forward_dense_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_dense(p, x))

#------------------------------------------------------------------------------------

def batch_query_moe(x_batches : list, moe : dict, func, remap_flag : bool = True):
    '''List of batched queries that will be passed through the model at once. Be aware of device OOM.'''
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])

    # Originally we used vmap here but this caused for large models always OOM on a T4, 
    # which is why we swapped to a on devcice sequential compute, forward passing the batch
    # through each expert.
    def run_in_batches(func, moe, x_batches):
        def step(carry, xb):
            yp = func(moe, xb).flatten()
            return carry, yp

        _, yp = jax.lax.scan(step, None, x_batched)
        return jnp.concatenate(yp, axis=0)

    yp = run_in_batches(func, moe, x_batched)
    yp_tail = func(moe, x_batches[-1]).flatten()
    yp = jnp.concatenate((yp, yp_tail), axis=0)

    # Remap
    yp_raw = yp.copy()
    if remap_flag:
        yp = remap(
            yp, 
            jnp.min(yp),
            jnp.max(yp),
            0,
            1,)

    return yp, yp_raw

#------------------------------------------------------------------------------------

def batch_query_moe_OOM(x_batches : list, moe : dict, func, at_once : int):
    '''List of batched queries that will be passed through the model by looping over subsets of batches to avoid OOM on device when passing all batches at once.
    \nPassing all batches at once can be done with batch_query_moe().'''

    # Forward batch subsets WITHOUT remapping the subsets already to 0.0 - 1.0
    yp_all = []
    for i in range(0, len(x_batches), at_once):
        yp, _ = batch_query_moe(x_batches[i:i+at_once], moe, func, remap_flag=False)
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

#------------------------------------------------------------------------------------

def export_moe( moe : dict, model_key : str) -> str:
    # Convert to NumPy and save parameters as dictionary of numpy arrays
    moe_gate = [np.array(l) for l in moe['gate']]
    moe_experts = [[np.array(l) for l in e] for e in moe['experts']]
    dimension = moe_gate[0].shape[0] - 1 #To account for bias

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250716_142015

    # Save layers as list of numpy arrays under gate or expert keyword.
    path = result_model_dir_registry[dimension] + "/" + timestamp + f"_{model_key}.npz"
    np.savez(
        path,
        gate = np.array(moe_gate, dtype=object),
        experts = np.array(moe_experts, dtype=object)
    )
    return path

def load_moe( path : str) -> dict:
    # Load moe .npz file 
    loaded = np.load(path, allow_pickle=True)
    
    # Unpack file and create core moe structure
    ## Simply joins into single dictionary and converts numpy arrays to JAX arrays
    moe = {}
    moe['experts'] = [jnp.array([l for l in e]) for e in loaded['experts']]
    moe['gate'] = [jnp.array(l) for l in loaded['gate']]

    return moe