import jax
import jax.numpy as jnp

from beh.adapter.shared import *
from beh.core.shared import *
from beh.registry import *
from beh.config_parser import *
from beh.core.moe import moe_forward_gate, moe_forward_expert

# For each batchsize we would like to use for sparse inference we need to define a below
# function block, since we want to JIT the inference and need to have static minibatch shapes
# during compile time. We could use jax.jit to pass static arguments, yet we can not pass 
# a jitted function as an argument to another jitted function. We would need to pass the 
# JIT minibatch function to the inference function as we require the gate indicies to
# divide the query into the minibatch.
# Could not think of a clean solution for this, but happy to hear good alternatives.
#------------------------------------------------------------------------------------

args, configs = parse_train()

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse_200K_175(p : dict, x : jax.Array):
    logits = moe_forward_gate(p['gate'], x)
    x, unpack_mini = minibatch_200K_175(x, logits)
    # Since PyTree hold layer with leading axis == number of experts and
    # we minibatch the queries for each expert, assuming uniformity we 
    # can vmap along PyTree and minibatched queries. Essential for sparsity.
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[unpack_mini[...,0], unpack_mini[...,1]]

@jax.jit
def moe_forward_sparse_INF_200K_175(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse_200K_175(p, x))

@jax.jit
def minibatch_200K_175(x : jax.Array , logits : jax.Array):
    # Constants from config
    nex = configs['moe175']['nex']
    in_dim = args.dim
    minibatch_size = int((200000 / nex ) * 1.1)

    # top-1 gating
    expert_idx = jnp.argmax(logits, axis=-1)    
    perm = jnp.argsort(expert_idx)
    x_sorted = x[perm]
    expert_idx_sorted = expert_idx[perm]

    # compute indiviual query index within each expert minibatch
    counts = jnp.bincount(expert_idx_sorted, length=nex)
    ## Start index of expert minibatch inside sorted query
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0], expert_idx_sorted.dtype), counts[:-1]]))
    minibatch_idx = jnp.arange(x.shape[0]) - starts[expert_idx_sorted]

    # Create padded minibatches and insert x using [expert index, minibatch index]
    padded_x = jnp.zeros((nex, minibatch_size, in_dim), x.dtype)
    padded_x = padded_x.at[expert_idx_sorted, minibatch_idx].set(x_sorted)

    # Create unpacking indicies from minibatch
    query_unsort = jnp.argsort(perm)
    unpack_mini = jnp.stack([expert_idx_sorted, minibatch_idx]).T[query_unsort]

    return padded_x, unpack_mini

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse_200K_48(p : dict, x : jax.Array):
    logits = moe_forward_gate(p['gate'], x)
    x, unpack_mini = minibatch_200K_48(x, logits)
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[unpack_mini[...,0], unpack_mini[...,1]]

@jax.jit
def moe_forward_sparse_INF_200K_48(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse_200K_48(p, x))

@jax.jit
def minibatch_200K_48(x : jax.Array , logits : jax.Array):
    nex = configs['moe48']['nex']
    in_dim = args.dim
    minibatch_size = int((200000 / nex ) * 1.1)
    expert_idx = jnp.argmax(logits, axis=-1)    
    perm = jnp.argsort(expert_idx)
    x_sorted = x[perm]
    expert_idx_sorted = expert_idx[perm]
    counts = jnp.bincount(expert_idx_sorted, length=nex)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0], expert_idx_sorted.dtype), counts[:-1]]))
    minibatch_idx = jnp.arange(x.shape[0]) - starts[expert_idx_sorted]
    padded_x = jnp.zeros((nex, minibatch_size, in_dim), x.dtype)
    padded_x = padded_x.at[expert_idx_sorted, minibatch_idx].set(x_sorted)
    query_unsort = jnp.argsort(perm)
    unpack_mini = jnp.stack([expert_idx_sorted, minibatch_idx]).T[query_unsort]

    return padded_x, unpack_mini

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse_200K_4(p : dict, x : jax.Array):
    logits = moe_forward_gate(p['gate'], x)
    x, unpack_mini = minibatch_200K_4(x, logits)
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[unpack_mini[...,0], unpack_mini[...,1]]

@jax.jit
def moe_forward_sparse_INF_200K_4(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse_200K_4(p, x))

@jax.jit
def minibatch_200K_4(x : jax.Array , logits : jax.Array):
    nex = configs['moe4']['nex']
    in_dim = args.dim
    minibatch_size = int((200000 / nex ) * 1.1)
    expert_idx = jnp.argmax(logits, axis=-1)    
    perm = jnp.argsort(expert_idx)
    x_sorted = x[perm]
    expert_idx_sorted = expert_idx[perm]
    counts = jnp.bincount(expert_idx_sorted, length=nex)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0], expert_idx_sorted.dtype), counts[:-1]]))
    minibatch_idx = jnp.arange(x.shape[0]) - starts[expert_idx_sorted]
    padded_x = jnp.zeros((nex, minibatch_size, in_dim), x.dtype)
    padded_x = padded_x.at[expert_idx_sorted, minibatch_idx].set(x_sorted)
    query_unsort = jnp.argsort(perm)
    unpack_mini = jnp.stack([expert_idx_sorted, minibatch_idx]).T[query_unsort]

    return padded_x, unpack_mini

#------------------------------------------------------------------------------------

# Create lookup table of sparse inference functions for each sparse MoE configuration
sparse_funcs_200K = {'moe175': moe_forward_sparse_INF_200K_175,
                'moe48': moe_forward_sparse_INF_200K_48,
                'moe4': moe_forward_sparse_INF_200K_4}

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse_2048_175(p : dict, x : jax.Array):
    logits = moe_forward_gate(p['gate'], x)
    x, unpack_mini = minibatch_2048_175(x, logits)
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[unpack_mini[...,0], unpack_mini[...,1]]

@jax.jit
def moe_forward_sparse_INF_2048_175(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse_2048_175(p, x))

@jax.jit
def minibatch_2048_175(x, logits):
    # Constants from config
    nex = configs['moe175']['nex']
    in_dim = args.dim
    minibatch_size = 2048

    expert_idx = jnp.argmax(logits, axis=-1)    
    perm = jnp.argsort(expert_idx)
    x_sorted = x[perm]
    expert_idx_sorted = expert_idx[perm]
    counts = jnp.bincount(expert_idx_sorted, length=nex)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0], expert_idx_sorted.dtype), counts[:-1]]))
    minibatch_idx = jnp.arange(x.shape[0]) - starts[expert_idx_sorted]
    padded_x = jnp.zeros((nex, minibatch_size, in_dim), x.dtype)
    padded_x = padded_x.at[expert_idx_sorted, minibatch_idx].set(x_sorted)
    query_unsort = jnp.argsort(perm)
    unpack_mini = jnp.stack([expert_idx_sorted, minibatch_idx]).T[query_unsort]
    return padded_x, unpack_mini

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse_2048_48(p : dict, x : jax.Array):
    logits = moe_forward_gate(p['gate'], x)
    x, unpack_mini = minibatch_2048_48(x, logits)
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[unpack_mini[...,0], unpack_mini[...,1]]

@jax.jit
def moe_forward_sparse_INF_2048_48(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse_2048_48(p, x))

@jax.jit
def minibatch_2048_48(x, logits):
    # Constants from config
    nex = configs['moe48']['nex']
    in_dim = args.dim
    minibatch_size = 2048

    expert_idx = jnp.argmax(logits, axis=-1)    
    perm = jnp.argsort(expert_idx)
    x_sorted = x[perm]
    expert_idx_sorted = expert_idx[perm]
    counts = jnp.bincount(expert_idx_sorted, length=nex)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0], expert_idx_sorted.dtype), counts[:-1]]))
    minibatch_idx = jnp.arange(x.shape[0]) - starts[expert_idx_sorted]
    padded_x = jnp.zeros((nex, minibatch_size, in_dim), x.dtype)
    padded_x = padded_x.at[expert_idx_sorted, minibatch_idx].set(x_sorted)
    query_unsort = jnp.argsort(perm)
    unpack_mini = jnp.stack([expert_idx_sorted, minibatch_idx]).T[query_unsort]
    return padded_x, unpack_mini

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse_2048_4(p : dict, x : jax.Array):
    logits = moe_forward_gate(p['gate'], x)
    x, unpack_mini = minibatch_2048_4(x, logits)
    out = jax.vmap(lambda p, x: moe_forward_expert(p, x))(p['experts'], x)
    return out[unpack_mini[...,0], unpack_mini[...,1]]

@jax.jit
def moe_forward_sparse_INF_2048_4(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse_2048_4(p, x))

@jax.jit
def minibatch_2048_4(x, logits):
    # Constants from config
    nex = configs['moe4']['nex']
    in_dim = args.dim
    minibatch_size = 2048

    expert_idx = jnp.argmax(logits, axis=-1)    
    perm = jnp.argsort(expert_idx)
    x_sorted = x[perm]
    expert_idx_sorted = expert_idx[perm]
    counts = jnp.bincount(expert_idx_sorted, length=nex)
    starts = jnp.cumsum(jnp.concatenate([jnp.array([0], expert_idx_sorted.dtype), counts[:-1]]))
    minibatch_idx = jnp.arange(x.shape[0]) - starts[expert_idx_sorted]
    padded_x = jnp.zeros((nex, minibatch_size, in_dim), x.dtype)
    padded_x = padded_x.at[expert_idx_sorted, minibatch_idx].set(x_sorted)
    query_unsort = jnp.argsort(perm)
    unpack_mini = jnp.stack([expert_idx_sorted, minibatch_idx]).T[query_unsort]
    return padded_x, unpack_mini

#------------------------------------------------------------------------------------

# Create lookup table of sparse inference functions for each sparse MoE configuration
sparse_funcs_2048 = {'moe175': moe_forward_sparse_INF_2048_175,
                'moe48': moe_forward_sparse_INF_2048_48,
                'moe4': moe_forward_sparse_INF_2048_4}
