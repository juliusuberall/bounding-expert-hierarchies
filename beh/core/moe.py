import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

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
    return jax.nn.tanh((x @ p[-1]) * 0.5) * 3.0

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
def moe_forward_dense_INF(p : dict, x : jax.Array, minibatch : jax.Array = None, expert_counter : jax.Array = None):
    return jax.nn.sigmoid(moe_forward_dense(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse(p : dict, x : jax.Array, minibatch : jax.Array, expert_counter : jax.Array):
    _ , idx = jax.lax.top_k(moe_forward_gate(p['gate'], x), 1)
    nex = jnp.array(p['experts'][0].shape[0], dtype=jnp.int32)
    x, result_idx1, result_idx2 = expert_mini_batch(idx.squeeze(), x, minibatch, expert_counter)

    # Since PyTree hold layer with leading axis == number of experts and
    # we minibatch the queries for each expert, assuming uniformity we 
    # can vmap along PyTree and minibatched queries. Essential for sparsity.
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[result_idx1, result_idx2].flatten()

@jax.jit
def moe_forward_sparse_INF(p : dict, x : jax.Array, minibatch : jax.Array, expert_counter : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse(p, x, minibatch, expert_counter))

#------------------------------------------------------------------------------------

@jax.jit
def expert_mini_batch(idx : jax.Array, x : jax.Array, minibatch : jax.Array, expert_counter : jax.Array):
    # Compute secondary indecies for placing in padded minibatch
    _ , idx2 = jax.lax.scan(body, expert_counter, idx)
    # Set in mini batch
    x = minibatch.at[idx, idx2].set(x)
    return x, idx, idx2

@jax.jit
def body(carry, x):
    # Scan through expert activation and compute 2nd dimension index
    counts = carry
    val_count = counts[x]
    counts = counts.at[x].add(1)
    return counts, val_count

def moe_forward_sparse_inf_PREP(moe : dict, batchsize : int, dimension : int):
    ## Define empty minibatch and expert counter that works with batchsize.
    ## Important and not definibale inside JIT as size depends on batchsize and nex.
    ## Used for minibatching queries for experts.
    nex = moe['experts'][0].shape[0]
    max_minibatch_size = int(batchsize / nex * 1.3) # Expecting uniformity +- some queries
    minibatch = jnp.zeros((nex, max_minibatch_size, dimension))
    expert_counter = jnp.zeros(nex, dtype=jnp.int32)
    return minibatch, expert_counter


#------------------------------------------------------------------------------------

def moe_error(x_batches : list, y : jax.Array , moe : dict, func, minibatch : jax.Array = None, expert_counter : jax.Array = None):

    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda x: func(moe, x, minibatch, expert_counter))(x_batched).flatten()
    ## Add tail remaining when batching with batch size
    x_tail = func(moe, x_batches[-1], minibatch, expert_counter)
    yp = jnp.concatenate((yp, x_tail.flatten()))

    # MSE
    yp_raw = yp.copy()
    yp = remap(
        yp, 
        jnp.min(yp),
        jnp.max(yp),
        0,
        1,)
    mse = jnp.mean((y - yp)**2)

    return mse, yp, yp_raw

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