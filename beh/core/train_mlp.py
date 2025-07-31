import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.mlp import *
from beh.core.params import *
from beh.core.registry import *
from beh.core.shared import *
from beh.core.benchmarking import *

from beh.styler.shared import *

def train_mlp(
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query : str,
    configs : dict,
    dimension : int):

    # Extract model configurations
    hid_lay = configs['mlp']['hidden_layer']

    # Set training hyperparameters
    epochs = configs['general']['epochs']
    batch_size = configs['general']['batch_size']
    learning_rate = configs['general']['learning_rate']
    threshold = configs['general']['boundary_threshold']
    loss_logging_frequency = configs['general']['loss_logging_frequency']

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

    @jax.jit
    def update(p, opt_state, xB, yB):
        grads = jax.grad(mlp_bce_loss)(p, xB, yB)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads

    # Training loop
    print(f"\nEpochs: {epochs} | Batch: {batch_size} | LearnRate: {learning_rate}")
    print(f"MLP: {mlp_arch}")
    print(f"+++++++++++++ Starting MLP training ++++++++++++++")
    val_loss_cache, fn_cache, fp_cache, epoch_cache = [], [], [], []
    for i in range(1, epochs + 1):

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs therefor much faster than jax
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=False)
        xB, yB = x[idx,...], y[idx,...]
        mlp, opt_state, gradient = update(mlp, opt_state, xB, yB)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            val_loss, yp  = mlp_error(x_batches, y, mlp)  
            val_loss_cache.append(val_loss) 

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(fn) 
            fp_cache.append(fp) 

            epoch_cache.append(i)     
            print(f"Epoch {i:05d}, Val-MSE-Loss: {round(float(val_loss),4):04f} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f}")
            checkpoint_mlp_export_plot_gradient(gradient, dimension, i)
    
    # Register training metrics
    model_key = 'mlp'
    reg_key = model_key + core_keys['train_val_loss_key']
    reg.add( reg_key, jnp.array(val_loss_cache))

    reg_key = model_key + core_keys['train_fn_key']
    reg.add( reg_key, jnp.array(fn_cache))

    reg_key = model_key + core_keys['train_fp_key']
    reg.add( reg_key, jnp.array(fp_cache))

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, jnp.array(epoch_cache))
    
    return mlp, reg