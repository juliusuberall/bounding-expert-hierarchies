import jax
import jax.numpy as jnp
import numpy as np
import optax

from datetime import datetime

from beh.adapter.shared import *
from beh.core.shared import *
from beh.registry import *
from beh.config_parser import *

# Get topk value from config
## We can not pass topk as argument due to JIT errors, 
## given that topk value determines the shape of output.
args, configs = parse_all()
topk = batch_size = configs['moe']['topk']

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_expert(p, x, idx):
    for e in p['experts'][:-1]:
        # We use the bias trick for all MoE implementation
        x = jnp.append(x, 1)
        x = jax.nn.tanh(jnp.dot(x, e[idx]))
    x = jnp.append(x, 1)
    return jax.nn.tanh(jnp.dot(x, p['experts'][-1][idx]) *0.5) *3.0

@jax.jit
def moe_forward_expert_INF(p, x, idx):
    return jax.nn.sigmoid(moe_forward_expert(p,x,idx))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_gate(p, x):
    for l in p['gate'][:-1]:
        x = jnp.append(x, 1)
        x = jax.nn.relu(jnp.dot(x, l))
    x = jnp.append(x, 1)
    return jax.nn.softmax(jnp.dot(x, p['gate'][-1]))

@jax.jit
def moe_forward_gate_INF(p, x):
    return moe_forward_gate(p, x)

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_dense(p, x):
    activation = moe_forward_gate(p, x)
    x = jax.vmap(lambda idx: moe_forward_expert(p, x, idx))(jnp.arange(activation.shape[0]))
    return jnp.sum(x * jnp.expand_dims(activation, axis=-1))

@jax.jit
def moe_forward_dense_INF(p, x):
    return jax.nn.sigmoid(moe_forward_dense(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse(p, x):
  _ , idx = jax.lax.top_k(moe_forward_gate(p, x), topk)
  return moe_forward_expert(p, x, idx)

@jax.jit
def moe_forward_sparse_INF(p, x):
    return jax.nn.sigmoid(moe_forward_sparse(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def moe_KL_BCE_loss(p, x, y): 
    # Binary Cross-Entropy 
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(yp, y))

    # KL-divergence of the expert activation distribution against uniform distribution 
    activation = (jax.vmap(lambda x: moe_forward_gate(p, x))(x)) * jnp.expand_dims(y,axis=1)
    g = 1/jnp.sum(y) * jnp.sum(activation, axis=0) 
    kl_loss = jnp.sum(g * jnp.log(g / (1 / activation.shape[1])))

    # Activation Entropy
    query_entropy = -jnp.sum(activation * jnp.log(activation + 1e-8), axis=1)
    ae_loss = jnp.mean(query_entropy)

    return kl_loss + bce_loss + ae_loss * 4.0

#------------------------------------------------------------------------------------

def moe_loss(x_batches : list, y : jax.Array , moe : dict, func):
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

def export_moe( moe : dict) -> str:
    # Convert to NumPy and save parameters as dictionary of numpy arrays
    moe_gate = [np.array(l) for l in moe['gate']]
    moe_experts = [[np.array(l) for l in e] for e in moe['experts']]
    dimensions = moe_gate[0].shape[0] - 1 #To account for bias

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250716_142015

    # Save layers as list of numpy arrays under gate or expert keyword.
    path = model_dir_registry[dimensions] + "/" + timestamp + f"_{dimensions}D_MoE.npz"
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

