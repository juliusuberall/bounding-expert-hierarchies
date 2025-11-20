import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.params import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.core.loss import moe_grid_train_loss
from beh.core.moe_grid import moe_grid_select, moe_grid_forward, batch_query_moe_grid
from beh.core.train_moe import expert_conservativness

from beh.styler.shared import *

#------------------------------------------------------------------------------------

def mask_grads(grads : list, frozen_ids : jax.Array):
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

#------------------------------------------------------------------------------------

def train_moe_grid(
    model_key : str,
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query : str,
    configs : dict,
    dimension : int):

    # Extract model configurations
    grid_dim = configs[model_key]['grid_dim']
    nex = grid_dim ** dimension
    expert_hid_lay = configs[model_key]['expert_hidden_layer']

    # Set training hyperparameters
    batch_size = configs['general']['batch_size']
    learning_rate = configs['general']['learning_rate']
    threshold = configs['general']['boundary_threshold']
    loss_logging_frequency = configs['general']['loss_logging_frequency']
    min_epochs = configs[model_key]['min_epochs']

    # Create validation loss batches
    x_batches = batch_data(x, batch_size)
    
    # Initalize MoE and optimizer
    expert_arch = [dimension] + expert_hid_lay + [1]
    moe_grid = init_experts(
        expert_arch,
        nex,
        key
    )
    opt = optax.adam(learning_rate)
    opt_state = opt.init(moe_grid)

    # Count total parameter and store
    total_p = count_parameter(expert_arch) * nex
    reg.add( model_key + core_keys['total_parameters_key'],
            total_p)
    reg.add( model_key + core_keys['active_parameters_key'],
            total_p)

    @jax.jit
    def update(p, opt_state, xB, yB, iB, negative_class_weight):
        grads = jax.grad(moe_grid_train_loss)(p, xB, yB, iB, negative_class_weight)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads
    
    # Training loop
    print(f"\nBatch: {batch_size} | LearnRate: {learning_rate}")
    print(f"{nex}x Experts: {expert_arch} | Total P: {total_p}")
    print(f"+++++++++++++ Starting {model_key} training ++++++++++++++")
    fn_cache, fp_cache, slope_cache, epoch_cache, con_experts_cache = [], [], [], [], []
    ## Initalize asymmetry factor for BCE
    negative_class_weight = jnp.array(1.0)

    i = 1
    fn = jnp.array(1.0)
    train_time_t0 = time.perf_counter_ns()
    making_conservative = False
    con_experts = jnp.array([])
    # Dont stop training until:
    # -> Min epochs trained
    # -> FN == 0
    while i < min_epochs or fn != 0.0:

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs much faster than jax for random sampling
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=True) # Replace=true makes this much faster and is okay in our case
        xB, yB = x[idx,...], y[idx,...]
        iB = moe_grid_select(xB, dimension, grid_dim)
        moe_grid, opt_state, gradient = update(moe_grid, opt_state, xB, yB, iB, negative_class_weight)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            yp , e_idx , yp_raw  = batch_query_moe_grid(x_batches, moe_grid, dimension, grid_dim)  

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(fn) 
            fp_cache.append(fp)
            slope_cache.append(fp)  

            # Track number of conservative experts
            con_experts = expert_conservativness(yp, y, e_idx, threshold, nex)
            con_experts_cache.append(con_experts.size / nex)

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f} | Con Experts: {con_experts.size}/{nex} | yp Range: {jnp.min(yp_raw):04f} to {jnp.max(yp_raw):04f}")
            checkpoint_moe_grid_export_plot_gradient(gradient, dimension, i)

            if i % min_epochs == 0 or making_conservative and len(slope_cache) == 10: 
                making_conservative = True
                slope_cache = []
                negative_class_weight /= 1.5
                print(f"Decreasing negative weight to {negative_class_weight}")

                # Change update function to freeze conservative experts.
                # Since all experts are part of each leaf as we store them combined we
                # have to do it like this to maintain speed.
                @jax.jit
                def update(p, opt_state, xB, yB, iB, negative_class_weight):
                    grads = jax.grad(moe_grid_train_loss)(p, xB, yB, iB, negative_class_weight)
                    # Freeze conservative experts
                    grads = mask_grads(grads, con_experts)
                    updates, opt_state = opt.update(grads, opt_state)
                    p = optax.apply_updates(p, updates)
                    return p, opt_state, grads
        i += 1
    
    # Register training metrics
    reg_key = model_key + core_keys['total_epochs']
    reg.add( reg_key, i)

    reg_key = model_key + core_keys['training_time']
    reg.add( reg_key, np.array((time.perf_counter_ns() - train_time_t0) / 1e9))

    reg_key = model_key + core_keys['train_fn_key']
    reg.add( reg_key, np.array(fn_cache))

    reg_key = model_key + core_keys['train_fp_key']
    reg.add( reg_key, np.array(fp_cache))

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, np.array(epoch_cache))

    reg_key = model_key + core_keys['train_conservative_experts_key']
    reg.add( reg_key, np.array(con_experts_cache))
    
    return moe_grid, reg
