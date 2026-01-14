import jax
import jax.numpy as jnp
import numpy as np

from beh.core.shared import positional_encoding
from beh.config_parser import parse_train

_ , configs = parse_train()

#------------------------------------------------------------------------------------

@jax.jit
def expert_forward_dense(p : list, x : jax.Array):
    x = positional_encoding(x, configs['general']['pe_num_freq'])
    for e in p[:-1]:
        # Bias Trick
        x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
        x = jax.nn.tanh(x @ e)
    x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    # We found that these additional factors improved the accurarcy as well as the "double" activation with tanh() and sigmoid() in the final layer.
    # The sigmoid is not defined here, because OPTAX does this for numerical stability in their BCE implementation. Therefore the regualr model output
    # needs to be additionally transformed with sigmoid()
    return jax.nn.tanh((x @ p[-1]) * 0.5) * 3.0 

@jax.jit
def expert_forward_dense_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(expert_forward_dense(p['experts'], x))

#------------------------------------------------------------------------------------

@jax.jit
def expert_forward_sparse(experts : list, x : jax.Array, idx : jax.Array):
    x = positional_encoding(x, configs['general']['pe_num_freq'])
    # Select parameters on per query basis
    for e in experts[:-1]:
        # Bias Trick
        x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
        x = jax.nn.tanh(jnp.einsum('bi,bij->bj', x, e[idx]))
    x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    out = jax.nn.tanh(jnp.einsum('bi,bij->bj', x, experts[-1][idx])* 0.5) * 3.0 
    return out

#------------------------------------------------------------------------------------

@jax.jit
def gate_forward(p : list, x : jax.Array):
    for l in p[:-1]:
        x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
        x = jax.nn.relu(x @ l)
    x = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    return jax.nn.softmax(x @ p[-1])

@jax.jit
def gate_forward_INF(p : dict, x : jax.Array):
    return gate_forward(p['gate'], x)

#====================================================================================

@jax.jit
def moe_forward_dense(p : dict, x : jax.Array):
    activation = gate_forward(p['gate'], x)
    y = jax.vmap(lambda p: expert_forward_dense(p, x), out_axes=1)(p['experts']).squeeze()
    return jnp.sum(y * activation, axis=-1)

@jax.jit
def moe_forward_dense_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_dense(p, x))

#------------------------------------------------------------------------------------

@jax.jit
def moe_forward_sparse(p : dict, x : jax.Array):
    logits = gate_forward(p['gate'], x)
    idx = jax.lax.top_k(logits, 1)[1].flatten()
    out = expert_forward_sparse(p['experts'], x, idx)
    return out

@jax.jit
def moe_forward_sparse_INF(p : dict, x : jax.Array):
    return jax.nn.sigmoid(moe_forward_sparse(p, x))

#====================================================================================

def batch_query_moe(x_batches : list, moe : dict, func, remap_flag : bool = True):
    '''List of batched queries that will be passed through the model at once. Be aware of device OOM.'''
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])

    # Originally we used vmap here but this caused for large models always OOM on a T4 with dense forward, 
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
    yp = np.concatenate((yp, yp_tail), axis=0)
    yp_raw = yp

    return yp, yp_raw

#------------------------------------------------------------------------------------

def batch_query_moe_OOM(x_batches : list, moe : dict, func, at_once : int) -> np.ndarray:
    '''List of batched queries that will be passed through the model by looping over subsets of batches to avoid OOM on device when passing all batches at once.
    \nPassing all batches at once can be done with batch_query_moe().'''

    yp_all = []
    for i in range(0, len(x_batches), at_once):
        yp, _ = batch_query_moe(x_batches[i:i+at_once], moe, func, remap_flag=False)
        yp_all.append(np.array(yp)) # unload from GPU
    yp_all = np.concatenate(yp_all, axis=0)
    
    return yp_all

#------------------------------------------------------------------------------------

def expert_conservativness(yp : np.ndarray, y : np.ndarray, e_idx : np.ndarray, threshold : float, nex : int) -> np.ndarray:
    '''Evaluate which of the experts is conservative and which not.
    \nReturns indicies of conservative experts which should be frozen.'''
    # Compute locations of FN
    fn_mask = (yp < threshold) * (y > 0)
    # Determine which experts caused FN
    fn_experts = e_idx[fn_mask]
    fn_experts = jnp.unique(fn_experts)
    conservative_experts = jnp.setdiff1d(jnp.arange(nex), fn_experts)
    return np.array(conservative_experts)

#------------------------------------------------------------------------------------

def mask_grads(grads, frozen_ids : jax.Array):
    '''Mask and freeze expert parameters based on passed id.'''
    if type(grads) == list:
        return mask_grads_moeg(grads, frozen_ids)
    elif type(grads) == dict:
        return mask_grads_moe(grads, frozen_ids)
    else:
        raise ValueError(f"Unsupported model type for expert freezing.")


def mask_grads_moe(grads : dict, frozen_ids : jax.Array):
    '''Freeze parameters of conservative experts.
    Could not think of an alternative because not each expert is an indidviual leaf in the PyTree.'''
    expert_grads = []
    for layer in grads["experts"]:
        mask = jnp.ones(layer.shape[0], dtype=layer.dtype)
        mask = mask.at[frozen_ids].set(0.0)   # 0 for frozen, 1 otherwise
        # Broadcast mask to match grads shape
        layer = layer * mask[:, None, None]
        expert_grads.append(layer)
    grads['experts'] = expert_grads
    return grads

def mask_grads_moeg(grads : list, frozen_ids : jax.Array):
    '''Freeze parameters of conservative experts.
    Could not think of an alternative because not each expert is an indidviual leaf in the PyTree.'''
    expert_grads = []
    for layer in grads:
        mask = jnp.ones(layer.shape[0], dtype=layer.dtype)
        mask = mask.at[frozen_ids].set(0.0)   # 0 for frozen, 1 otherwise
        # Broadcast mask to match grads shape
        layer = layer * mask[:, None, None]
        expert_grads.append(layer)
    return expert_grads