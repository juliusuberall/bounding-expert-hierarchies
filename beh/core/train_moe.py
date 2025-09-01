import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.moe import *
from beh.core.params import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.core.loss import moe_train_loss, moe_train_loss_1_expert

from beh.styler.shared import *

def train_moe(
    model_key : str,
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query : str,
    configs : dict,
    dimension : int):

    # Extract model configurations
    nex = configs[model_key]['nex']
    gate_hid_lay = configs[model_key]['gate_hidden_layer']
    expert_hid_lay = configs[model_key]['expert_hidden_layer']

    # Set training hyperparameters
    batch_size = configs['general']['batch_size']
    learning_rate = configs['general']['learning_rate']
    threshold = configs['general']['boundary_threshold']
    loss_logging_frequency = configs['general']['loss_logging_frequency']
    fp_slope_thresh = configs['general']['fp_slope_stop']

    # Create validation loss batches
    x_batches = batch_data(x, batch_size)
    
    # Initalize MoE and optimizer
    gate_arch = [dimension] + gate_hid_lay + [nex]
    expert_arch = [dimension] + expert_hid_lay + [1]
    moe = init_moe(
        gate_arch,
        expert_arch,
        nex,
        key
    )
    opt = optax.adam(learning_rate)
    opt_state = opt.init(moe)

    # Count total and active parameter and store
    ## Dense MoE
    dkey = f'{model_key}_dense'
    total_p = count_parameter(expert_arch) * nex + count_parameter(gate_arch)
    reg.add( dkey + core_keys['total_parameters_key'],
            total_p)
    reg.add( dkey + core_keys['active_parameters_key'],
            total_p)
    ## Sparse MoE
    skey = f'{model_key}_sparse'
    active_p = count_parameter(expert_arch) + count_parameter(gate_arch)
    reg.add( skey + core_keys['total_parameters_key'],
            total_p)
    reg.add( skey + core_keys['active_parameters_key'],
            active_p)
    
    if nex == 1:
        @jax.jit
        def update(p, opt_state, xB, yB, negative_class_weight):
            grads = jax.grad(moe_train_loss_1_expert)(p, xB, yB, negative_class_weight)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state, grads
    else:
        @jax.jit
        def update(p, opt_state, xB, yB, negative_class_weight):
            grads = jax.grad(moe_train_loss)(p, xB, yB, negative_class_weight)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state, grads

    # Training loop
    print(f"\nBatch: {batch_size} | LearnRate: {learning_rate}")
    print(f"Gate: {gate_arch} | {nex}x Experts: {expert_arch} | Total P: {total_p}")
    print(f"+++++++++++++ Starting {model_key} training ++++++++++++++")
    val_loss_cache, fn_cache, fp_cache, slope_cache, confidence_cache, epoch_cache = [], [], [], [], [], []
    ## Initalize self balancing factor for BCE
    negative_class_weight = jnp.array(1.0)
    
    i = 1
    fn = jnp.array(1.0)
    fp_slope = jnp.array(1.0)
    train_time_t0 = time.perf_counter_ns()
    # Dont stop training until:
    # -> Min epochs trained
    # -> FN == 0
    # -> FP plateaus
    while fn != 0.0 or jnp.abs(fp_slope) > fp_slope_thresh:

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs therefor much faster than jax
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=True) # Replace true makes this much faster
        xB, yB = x[idx,...], y[idx,...]
        moe, opt_state, gradient = update(moe, opt_state, xB, yB, negative_class_weight)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            val_loss, yp , __  = moe_error(x_batches, y, moe, moe_forward_dense_INF)  
            val_loss_cache.append(val_loss) 

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(fn) 
            fp_cache.append(fp)
            slope_cache.append(fp)  

            # Compute False-Positive slope
            fp_slope = fp - jnp.mean(jnp.array(slope_cache[-5:]))

            # Confidence
            confidence , _ , _ = gating_confidence(moe=moe, x_batches=x_batches)
            confidence_cache.append(confidence)

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d}, Val-MSE-Loss: {round(float(val_loss),4):04f} | Confidence: {round(float(confidence),4):04f} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f} | FP-slope: {round(float(fp_slope),4):04f}")
            checkpoint_moe_export_plot_gradient(gradient, dimension, i)

            if i > 2000 and  fp_slope < 0 and fp_slope > -fp_slope_thresh and len(slope_cache) >= 5: 
                slope_cache = []
                negative_class_weight /= 2
                print(f"Decreasing negative weight to {negative_class_weight}")
        i += 1

    
    # Register training metrics
    reg_key = model_key + core_keys['total_epochs']
    reg.add( reg_key, i)

    reg_key = model_key + core_keys['training_time']
    reg.add( reg_key, jnp.array((time.perf_counter_ns() - train_time_t0) / 1e9))

    reg_key = model_key + core_keys['train_val_loss_key']
    reg.add( reg_key, jnp.array(val_loss_cache))

    reg_key = model_key + core_keys['train_fn_key']
    reg.add( reg_key, jnp.array(fn_cache))

    reg_key = model_key + core_keys['train_fp_key']
    reg.add( reg_key, jnp.array(fp_cache))

    reg_key = model_key + core_keys['train_confidence_key']
    reg.add( reg_key, jnp.array(confidence_cache))

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, jnp.array(epoch_cache))
    
    return moe, reg