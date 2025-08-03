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

# Get topk value from config
## We can not pass topk as argument due to JIT errors, 
## given that topk value determines the shape of output.
args, configs = parse_all()
topk = configs['general']['topk']

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
def moe_train_loss(p, x, y):
    epsilon = 1e-8

    # Binary Cross-Entropy 
    yp = jax.vmap(lambda x: moe_forward_dense(p,x))(x)
    bce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(yp, y))

    # KL-divergence of the expert activation distribution against uniform distribution 
    activation = (jax.vmap(lambda x: moe_forward_gate(p, x))(x)) * jnp.expand_dims(y,axis=1)
    g = 1/jnp.sum(y) * jnp.sum(activation, axis=0) 
    kl_loss = jnp.sum(g * jnp.log(g / (1 / activation.shape[1])))

    # Gate Activation Entropy
    query_entropy = -jnp.sum(activation * jnp.log(activation + epsilon), axis=1)
    ae_loss = jnp.mean(query_entropy)

    return kl_loss + bce_loss + ae_loss * 4.0

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

def moe_min_inference_speed(
        x_batches : jax.Array,
        moe : dict, 
        func, 
        iterations : int,
        iteration_size : int,
        configs : dict):
    '''
    Measures the minimum inference speed of a MoE for N batched queries, averaged over M iterations.
    \nRandomly creates batches from the data provided to this function.
    '''
    # Generla batch size is defined for all models of same dimensionalty 
    batch_size = configs['general']['batch_size']

    x = jnp.array(x_batches[:-1]).reshape((-1,2))
    min_speed = []
    for i in range(iterations):
        speed = []
        for j in range(iteration_size):
            xj = x[np.random.choice(x.shape[0], batch_size)]

            # Measure inference
            start = time.perf_counter()
            jax.vmap(lambda x: func(moe, x))(xj).block_until_ready()
            end = time.perf_counter()

            speed.append((end - start) * 1e6)  # convert to microseconds
        min_speed.append(np.min(speed))
        end = '\n' if i == (iterations - 1) else '\r'
        print(f'{i+1}/{iterations} iterations with {iteration_size} repeated queries of batch size {batch_size} -> {func.__name__}()', end = end, flush = True)
    
    # Averages the minimum measures from all iterations
    min_speed = jnp.mean(jnp.array(min_speed))

    return min_speed

#------------------------------------------------------------------------------------

def export_moe( moe : dict, model_key : str) -> str:
    # Convert to NumPy and save parameters as dictionary of numpy arrays
    moe_gate = [np.array(l) for l in moe['gate']]
    moe_experts = [[np.array(l) for l in e] for e in moe['experts']]
    dimension = moe_gate[0].shape[0] - 1 #To account for bias

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250716_142015

    # Save layers as list of numpy arrays under gate or expert keyword.
    path = model_dir_registry[dimension] + "/" + timestamp + f"_{model_key}.npz"
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

