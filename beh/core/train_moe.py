import jax
import jax.numpy as jnp
import numpy as np
import optax

from beh.core.moe import *
from beh.core.params import *
from beh.styler.shared import *
from beh.core.registry import *

def train_moe(
    key : jax.Array,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    query : str,
    configs,
    dimension : int):

    # Extract model configurations
    nex = configs['moe']['nex']
    gate_hid_lay = configs['moe']['gate_hidden_layer']
    expert_hid_lay = configs['moe']['expert_hidden_layer']

    # Set training hyperparameters
    epoch = 0
    epochs = configs['general']['epochs']
    batch_size = configs['general']['batch_size']
    learning_rate = configs['general']['learning_rate']
    loss_logging_frequency = configs['general']['loss_logging_frequency']

    # Create validation loss batches
    batches = batch_data(x, batch_size)
    
    # Initalize MoE and optimizer
    moe = init_moe(
        [dimension] + gate_hid_lay + [nex],
        [dimension] + expert_hid_lay + [1],
        nex,
        key
    )
    opt = optax.adam(learning_rate)
    opt_state = opt.init(moe)

    @jax.jit
    def update(p, opt_state, xB, yB):
        grads = jax.grad(moe_KL_BCE_loss)(p, xB, yB)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads

    # Training loop
    print(f"++++++++++++ Starting MoE training ++++++++++++")
    val_loss_cache = [loss_logging_frequency]
    for i in range(epochs):

        # Used numpy for random sampling because we dont need random determinism
        # and numpy runs therefor much faster on CPU when debugging 
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=False)
        xB, yB = x[idx,...], y[idx,...]
        moe, opt_state, gradient = update(moe, opt_state, xB, yB)
        
        if epoch % loss_logging_frequency == 0: 
            val_loss = moe_loss_dense(batches, y, moe)  
            val_loss_cache.append(val_loss)          
            print(f"Epoch {epoch}, Val-MSE-Loss: {val_loss}")
            checkpoint_moe_export_plot_gradient(gradient, dimension, epoch)

        epoch += 1
    
    reg_key = 'moe' + core_registration_keys['train_val_loss_key']
    reg.add( reg_key, jnp.array(val_loss_cache))
    
    return moe, reg