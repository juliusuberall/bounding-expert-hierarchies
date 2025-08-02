import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.moe import *
from beh.core.params import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *

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
    nex = configs['moe']['nex']
    gate_hid_lay = configs['moe']['gate_hidden_layer']
    expert_hid_lay = configs['moe']['expert_hidden_layer']

    # Set training hyperparameters
    epochs = configs['general']['epochs']
    batch_size = configs['general']['batch_size']
    learning_rate = configs['general']['learning_rate']
    threshold = configs['general']['boundary_threshold']
    loss_logging_frequency = configs['general']['loss_logging_frequency']

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

    @jax.jit
    def update(p, opt_state, xB, yB):
        grads = jax.grad(moe_train_loss)(p, xB, yB)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads

    # Training loop
    print(f"\nEpochs: {epochs} | Batch: {batch_size} | LearnRate: {learning_rate}")
    print(f"Gate: {gate_arch} | {nex}x Experts: {expert_arch}")
    print(f"+++++++++++++ Starting {model_key} training ++++++++++++++")
    val_loss_cache, fn_cache, fp_cache, confidence_cache, epoch_cache = [], [], [], [], []
    for i in range(1, epochs + 1):

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs therefor much faster than jax
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=False)
        xB, yB = x[idx,...], y[idx,...]
        moe, opt_state, gradient = update(moe, opt_state, xB, yB)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            val_loss, yp , __  = moe_error(x_batches, y, moe, moe_forward_dense_INF)  
            val_loss_cache.append(val_loss) 

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(fn) 
            fp_cache.append(fp) 

            # Confidence
            confidence , _ , _ = gating_confidence(moe=moe, x_batches=x_batches, reg=reg)
            confidence_cache.append(confidence)

            epoch_cache.append(i)     
            print(f"Epoch {i:05d}, Val-MSE-Loss: {round(float(val_loss),4):04f} | Confidence: {round(float(confidence),4):04f} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f}")
            checkpoint_moe_export_plot_gradient(gradient, dimension, i)
    
    # Register training metrics
    model_key = f'{model_key}_{dimension}'
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