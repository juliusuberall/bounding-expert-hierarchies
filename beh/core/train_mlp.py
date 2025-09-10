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
    min_epochs = configs[model_key]['min_epochs']

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
    val_loss_cache, fn_cache, fp_cache, slope_cache, epoch_cache = [], [], [], [], []
    ## Initalize self balancing factor for BCE
    negative_class_weight = jnp.array(1.0)

    i = 1
    fn = jnp.array(1.0)
    fp_slope = jnp.array(1.0)
    train_time_t0 = time.perf_counter_ns()
    making_conservative = False
    # Dont stop training until:
    # -> Min epochs trained
    # -> FN == 0
    # -> FP plateaus
    while i < min_epochs or fn != 0.0:

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
            slope_cache.append(fp) 

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d}, Val-MSE-Loss: {round(float(val_loss),4):04f} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f}")
            checkpoint_mlp_export_plot_gradient(gradient, dimension, i)
        
            if i % min_epochs == 0 or making_conservative and len(slope_cache) == 10: 
                making_conservative = True
                negative_class_weight /= 4
                slope_cache = []
                print(f"Decreasing negative weight to {negative_class_weight}")
        i += 1
    
    # Register training metrics
    reg_key = model_key + core_keys['total_epochs']
    reg.add( reg_key, i)

    reg_key = model_key + core_keys['training_time']
    reg.add( reg_key, np.array((time.perf_counter_ns() - train_time_t0) / 1e9))

    reg_key = model_key + core_keys['train_val_loss_key']
    reg.add( reg_key, np.array(val_loss_cache))

    reg_key = model_key + core_keys['train_fn_key']
    reg.add( reg_key, np.array(fn_cache))

    reg_key = model_key + core_keys['train_fp_key']
    reg.add( reg_key, np.array(fp_cache))

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, np.array(epoch_cache))
    
    return mlp, reg