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
from beh.styler.dim2 import export_plot_2D_moe_internal

#------------------------------------------------------------------------------------

# Evaluate for each expert if conservative
def expert_conservativness(yp : jax.Array, y : jax.Array, e_idx : jax.Array, threshold : float, nex : int):
    '''Evaluate which of the experts is conservative and which not.
    \nReturns indicies of conservative experts which should be frozen.'''
    # Compute locations of FN
    fn_mask = (yp < threshold) * (y > 0)
    # Determine which experts caused FN
    fn_experts = e_idx[fn_mask]
    fn_experts = jnp.unique(fn_experts)
    conservative_experts = jnp.setdiff1d(jnp.arange(nex), fn_experts)
    return conservative_experts

#------------------------------------------------------------------------------------

def mask_grads(grads : dict, frozen_ids : jax.Array):
    '''Freeze parameters of conservative experts by setting the zero to 0 at those.
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

#------------------------------------------------------------------------------------

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
    min_epochs = configs[model_key]['min_epochs']

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
    def update(p, opt_state, xB, yB, negative_class_weight):
        grads = jax.grad(moe_train_loss)(p, xB, yB, negative_class_weight)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, grads

    # Training loop
    print(f"\nBatch: {batch_size} | LearnRate: {learning_rate}")
    print(f"Gate: {gate_arch} | {nex}x Experts: {expert_arch} | Total P: {total_p}")
    print(f"+++++++++++++ Starting {model_key} training ++++++++++++++")
    fn_cache, fp_cache, slope_cache, confidence_cache, epoch_cache, active_e_cache = [], [], [], [], [], []
    ## Initalize self balancing factor for BCE
    negative_class_weight = jnp.array(1.0)
    
    i = 1
    fn = jnp.array(1.0)
    train_time_t0 = time.perf_counter_ns()
    making_conservative = False
    flip = 0
    # Dont stop training until:
    # -> Min epochs trained
    # -> FN == 0
    # -> FP plateaus
    while i < min_epochs or fn != 0.0:

        # Using numpy for random sampling because we dont need random
        # determinism and numpy runs therefor much faster than jax
        idx = np.random.choice(np.arange(x.shape[0]),batch_size,replace=True) # Replace true makes this much faster
        xB, yB = x[idx,...], y[idx,...]
        moe, opt_state, gradient = update(moe, opt_state, xB, yB, negative_class_weight)
        
        if i % loss_logging_frequency == 0: 
            # Error
            ## Should be ideally over all data, otherwise conservativness calculation needs to be reworked
            yp , __  = batch_query_moe(x_batches, moe, sparse_funcs_2048[model_key])  

            # False-Negatives and False-Positives 
            fn, fp = get_fn_fp_rate(yp, y, threshold = threshold)
            fn_cache.append(fn) 
            fp_cache.append(fp)
            slope_cache.append(fp)  

            # Confidence
            confidence , _ , e_idx = gating_confidence(moe=moe, x_batches=x_batches)
            confidence_cache.append(confidence)
            active_e = jnp.unique(e_idx.flatten()).size
            active_e_cache.append(active_e / nex)

            # Print epoch stats
            epoch_cache.append(i)     
            print(f"Epoch {i:05d} | Confidence: {round(float(confidence),4):04f} | Sparse FN: {round(float(fn),4):04f} | Sparse FP: {round(float(fp),4):04f} | Active Experts: {active_e}/{nex}")
            checkpoint_moe_export_plot_gradient(gradient, dimension, i)

            if i % min_epochs == 0 or making_conservative and len(slope_cache) == 10: 
                making_conservative = True
                slope_cache = []
                flip += 1

                # Switch back and forth between optimzing either experts or gate 
                if flip % 2 == 0:
                    print('Adjust Gate')
                    # Freeze gate with multi_transform and remove Adam decay on gradient.
                    ## Makes the gate gradient strictly 0 during updates
                    tx = optax.multi_transform(
                        {
                            "e": optax.set_to_zero(),
                            "g": optax.adam(0.0001)
                        },
                        param_labels={"experts": "e", "gate": "g"}
                    )
                    opt = tx
                    opt_state = opt.init(moe)

                    # Change update function to freeze conservative experts.
                    # Since all experts are part of each leaf as we store them combined we
                    # have to do it like this to maintain speed.
                    @jax.jit
                    def update(p, opt_state, xB, yB, negative_class_weight):
                        grads = jax.grad(moe_train_loss)(p, xB, yB, negative_class_weight)
                        grads = mask_grads(grads, con_experts)
                        updates, opt_state = opt.update(grads, opt_state)
                        p = optax.apply_updates(p, updates)
                        return p, opt_state, grads
                
                else:
                    print('Adjust Experts')
                    negative_class_weight /= 1.5
                    # Determine which experts to freeze because they are conservative
                    con_experts = expert_conservativness(yp, y, e_idx, threshold, nex)

                    # Freeze gate with multi_transform and remove Adam decay on gradient.
                    ## Makes the gate gradient strictly 0 during updates
                    tx = optax.multi_transform(
                        {
                            "e": optax.adam(0.0001),
                            "g": optax.set_to_zero()
                        },
                        param_labels={"experts": "e", "gate": "g"}
                    )
                    opt = tx
                    opt_state = opt.init(moe)

                    # Change update function to freeze conservative experts.
                    # Since all experts are part of each leaf as we store them combined we
                    # have to do it like this to maintain speed.
                    @jax.jit
                    def update(p, opt_state, xB, yB, negative_class_weight):
                        grads = jax.grad(moe_train_loss)(p, xB, yB, negative_class_weight)
                        grads = mask_grads(grads, con_experts)
                        updates, opt_state = opt.update(grads, opt_state)
                        p = optax.apply_updates(p, updates)
                        return p, opt_state, grads

                print(f"{con_experts.size}/{active_e} experts are conservative\nDecreasing negative weight to {negative_class_weight}")
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

    reg_key = model_key + core_keys['train_confidence_key']
    reg.add( reg_key, np.array(confidence_cache))

    reg_key = model_key + core_keys['train_epoch_key']
    reg.add( reg_key, np.array(epoch_cache))

    reg_key = model_key + core_keys['active_experts_key']
    reg.add( reg_key, np.array(active_e_cache))
    
    return moe, reg