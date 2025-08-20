import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.moe import *
from beh.core.params import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.core.loss import moe_train_loss

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

    @jax.jit
    def update(p, opt_state, xB, yB, self_balance):
        grads = jax.grad(moe_train_loss)(p, xB, yB, self_balance)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads

    # Training loop
    print(f"\nEpochs: {epochs} | Batch: {batch_size} | LearnRate: {learning_rate}")
    print(f"Gate: {gate_arch} | {nex}x Experts: {expert_arch} | Total P: {total_p}")
    print(f"+++++++++++++ Starting {model_key} training ++++++++++++++")
    val_loss_cache, fn_cache, fp_cache, confidence_cache, epoch_cache = [], [], [], [], []
    ## Initalize self balancing factor for BCE
    self_balance = jnp.array(0.0)
    self_balance_steps = 1 / epochs
    
    train_time_t0 = time.perf_counter_ns()
    for i in range(1, epochs + 1):

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs therefor much faster than jax
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=True) # Replace true makes this much faster
        xB, yB = x[idx,...], y[idx,...]
        moe, opt_state, gradient = update(moe, opt_state, xB, yB, self_balance)
        self_balance += self_balance_steps
        
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
            confidence , _ , _ = gating_confidence(moe=moe, x_batches=x_batches)
            confidence_cache.append(confidence)

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d}, Val-MSE-Loss: {round(float(val_loss),4):04f} | Confidence: {round(float(confidence),4):04f} | FN: {round(float(fn),4):04f} | FP: {round(float(fp),4):04f}")
            checkpoint_moe_export_plot_gradient(gradient, dimension, i)
    
    # Register training metrics
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