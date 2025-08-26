import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.mlp import *
from beh.core.params import *
from beh.core.registry import *
from beh.core.shared import *
from beh.core.benchmarking import *
from beh.core.loss import mlp_bce_loss

from beh.styler.shared import *

def train_mlp(
    model_key : str,
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query : str,
    configs : dict,
    dimension : int):

    # Extract model configurations
    hid_lay = configs[model_key]['hidden_layer']

    # Set training hyperparameters
    batch_size = configs['general']['batch_size']
    learning_rate = configs['general']['learning_rate']
    threshold = configs['general']['boundary_threshold']
    loss_logging_frequency = configs['general']['loss_logging_frequency']
    fp_slope_thresh = configs['general']['fp_slope_stop']

    # Create validation loss batches
    x_batches = batch_data(x, batch_size)
    
    # Initalize MLP and optimizer
    mlp_arch = [dimension] + hid_lay + [1]
    mlp = init_layer(
        mlp_arch,
        key
    )
    opt = optax.adam(learning_rate)
    opt_state = opt.init(mlp)

    # Count total parameter and store
    total_p = count_parameter(mlp_arch)
    reg.add( model_key + core_keys['total_parameters_key'],
            total_p)
    reg.add( model_key + core_keys['active_parameters_key'],
            total_p)

    @jax.jit
    def update(p, opt_state, xB, yB, negative_class_weight):
        grads = jax.grad(mlp_bce_loss)(p, xB, yB, negative_class_weight)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads

    # Training loop
    print(f"\nBatch: {batch_size} | LearnRate: {learning_rate}")
    print(f"MLP: {mlp_arch} | Total P: {total_p}")
    print(f"+++++++++++++ Starting {model_key} training ++++++++++++++")
    val_loss_cache, fn_cache, fp_cache, epoch_cache = [], [], [], []
    ## Initalize self balancing factor for BCE
    negative_class_weight = jnp.array(1.0)

    i = 1
    fn = jnp.array(1.0)
    fp_slope = jnp.array(1.0)
    train_time_t0 = time.perf_counter_ns()
    # Dont stop training until:
    # -> FN == 0
    # -> FP plateaus
    while fn != 0.0 or fp_slope > 0 or fp_slope < fp_slope_thresh:

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs therefor much faster than jax
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=True) # Replace true makes this much faster
        xB, yB = x[idx,...], y[idx,...]
        mlp, opt_state, gradient = update(mlp, opt_state, xB, yB, negative_class_weight)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            val_loss, yp  = mlp_error(x_batches, y, mlp)  
            val_loss_cache.append(val_loss) 

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(fn) 
            fp_cache.append(fp) 

            # Compute False-Positive slope
            fp_slope = fp - jnp.mean(jnp.array(fp_cache[-10:]))

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d}, Val-MSE-Loss: {round(float(val_loss),4):04f} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f}")
            checkpoint_mlp_export_plot_gradient(gradient, dimension, i)
        
            if fp_slope < 0 and fp_slope > fp_slope_thresh and i > 1000: 
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

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, jnp.array(epoch_cache))
    
    return mlp, reg