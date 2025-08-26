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
def moe_forward_expert(p : dict, x : jax.Array, idx):
    for e in p['experts'][:-1]:
        # Bias Trick
        x = jnp.append(x, 1)
        x = jax.nn.tanh(jnp.dot(x, e[idx]))
    x = jnp.append(x, 1)
    return jax.nn.tanh(jnp.dot(x, p['experts'][-1][idx]) * 0.5) * 3.0

@jax.jit
def moe_forward_expert_INF(p : dict, x : jax.Array, idx):
    return jax.nn.sigmoid(moe_forward_expert(p,x,idx))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_gate(p : dict, x : jax.Array):
    for l in p['gate'][:-1]:
        x = jnp.append(x, 1)
        x = jax.nn.relu(jnp.dot(x, l))
    x = jnp.append(x, 1)
    return jax.nn.softmax(jnp.dot(x, p['gate'][-1]))

@jax.jit
def moe_forward_gate_INF(p : dict, x : jax.Array):
    return moe_forward_gate(p, x)

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_dense(p : dict, x : jax.Array):
    activation = moe_forward_gate(p, x)
    x = jax.vmap(lambda idx: moe_forward_expert(p, x, idx))(jnp.arange(activation.shape[0]))
    return jnp.sum(x * jnp.expand_dims(activation, axis=-1))

@jax.jit
def moe_forward_dense_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_dense(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse(p : dict, x : jax.Array):
  _ , idx = jax.lax.top_k(moe_forward_gate(p, x), 1)
  return moe_forward_expert(p, x, idx)

@jax.jit
def moe_forward_sparse_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse(p, x))

#------------------------------------------------------------------------------------

def moe_error(x_batches : list, y : jax.Array , moe : dict, func):
    ## Trim tail of x that does not fit with batchsize
    ## Minibatching of shape [batch, minibatch, coordinate]
    ## -> Causes double nested vmapping
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda batch: 
                  jax.vmap(lambda x: func(moe, x))(batch)
                  )(x_batched).flatten()

    ## Add tail remaining when batching with batch size
    x_tail = jax.vmap(lambda x: func(moe,x))(x_batches[-1])
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