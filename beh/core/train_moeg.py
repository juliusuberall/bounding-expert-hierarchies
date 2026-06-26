import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.params import *
from beh.core.registry import CoreRegistry
from beh.core.moe_benchmarking import *
from beh.core.shared import pe_dim
from beh.core.loss import moeg_train_loss
from beh.core.moeg import moeg_select, batch_query_moeg
from beh.core.train_moe import expert_conservativness

from beh.styler.shared import *

#------------------------------------------------------------------------------------

def train_moeg(
    model_key : str,
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query_dim : int,
    configs : dict,
    dimension : int):

    # Extract model configurations
    grid_dim = configs[model_key]['grid_dim']
    nex = grid_dim ** dimension
    expert_hid_lay = configs[model_key]['expert_hidden_layer']

    # Set training hyperparameters
    batch_size = configs['general']['batch_size']
    pe_num_freq = configs['general']['pe_num_freq']
    learning_rate = configs['general']['learning_rate']
    learning_rate_con = configs['general']['learning_rate_conservative']
    threshold = configs['general']['boundary_threshold']
    loss_logging_frequency = configs['general']['loss_logging_frequency']
    min_epochs = configs[model_key]['min_epochs']
    gammas = configs['general']['gamma']
    gamma = gammas[model_key[-1]]

    # Create validation loss batches
    x_batches = batch_data(x, batch_size)
    
    # Initalize MoE and optimizer
    pe_query_dim = pe_dim(query_dim, pe_num_freq)
    expert_arch = [pe_query_dim] + expert_hid_lay + [1]
    moeg = init_experts(
        expert_arch,
        nex,
        key
    )
    opt = optax.adam(learning_rate)
    opt_state = opt.init(moeg)

    # Count total parameter and store
    total_p = count_parameter(expert_arch) * nex
    active_p = count_parameter(expert_arch)
    reg.add( model_key + core_keys['total_parameters_key'], total_p)
    reg.add( model_key + core_keys['active_parameters_key'], active_p)

    # Define model training update
    @jax.jit
    def update(p, opt_state, xB, yB, iB, negative_class_weight):
        grads = jax.grad(moeg_train_loss)(p, xB, yB, iB, negative_class_weight, gamma)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads
    
    # Training loop
    print(f"\nBatchsize: {batch_size} | LearnRate: {learning_rate}")
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
        iB = moeg_select(xB, moeg)
        moeg, opt_state, gradient = update(moeg, opt_state, xB, yB, iB, negative_class_weight)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            yp , e_idx  = batch_query_moeg(x_batches, moeg)  

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(float(fn)) 
            fp_cache.append(float(fp))
            slope_cache.append(fp)  

            # Track number of conservative experts
            con_experts = expert_conservativness(yp, y, e_idx, threshold, nex)
            con_experts_cache.append(float(con_experts.size / nex))

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f} | Con Experts: {con_experts.size}/{nex} | yp Range: {jnp.min(yp):04f} to {jnp.max(yp):04f}")
            checkpoint_moeg_export_plot_gradient(gradient, dimension, i)

            if i % min_epochs == 0 or making_conservative and len(slope_cache) == 10: 
                if i == min_epochs:
                    opt = optax.adam(learning_rate_con)
                    opt_state = opt.init(moeg)
                    print(f'Learning rate changed to: {learning_rate_con}')

                making_conservative = True
                slope_cache = []
                negative_class_weight /= 1.5
                print(f"Decreasing negative weight to {negative_class_weight}")

                # Change update function to freeze conservative experts.
                # Since all experts are part of each leaf as we store them combined we
                # have to do it like this to maintain speed.
                @jax.jit
                def update(p, opt_state, xB, yB, iB, negative_class_weight):
                    grads = jax.grad(moeg_train_loss)(p, xB, yB, iB, negative_class_weight, gamma)
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
    reg.add( reg_key, np.array((time.perf_counter_ns() - train_time_t0) / 6e10))

    reg_key = model_key + core_keys['train_fn_key']
    reg.add( reg_key, np.array(fn_cache))

    reg_key = model_key + core_keys['train_fp_key']
    reg.add( reg_key, np.array(fp_cache))

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, np.array(epoch_cache))

    reg_key = model_key + core_keys['train_conservative_experts_key']
    reg.add( reg_key, np.array(con_experts_cache))

    reg_key = model_key + core_keys['architecture']
    reg.add( reg_key, f"{nex}x E {str(expert_arch).replace(" ", "")}")
    
    return moeg, reg
