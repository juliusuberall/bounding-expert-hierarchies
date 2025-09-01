import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beh.registry import *
from beh.core.registry import *
from beh.core.params import count_parameter

# Plot the gradient of a neural network training
def checkpoint_moe_export_plot_gradient(gradient, dimension, epoch):
    '''Visualize the MoE gradient during training and export as plot.'''
  
    # Create plot
    plt.title(f"Learning Signal {dimension}D MoE | Epoch {epoch}")
    for l in gradient['gate']:
        plt.plot(l.flatten(), linewidth=0.5)
        v, i = jax.lax.top_k(l.flatten(), 1)
        plt.scatter(i,v,label='GL')
    for l in gradient['experts']:
        plt.plot(l.flatten(), linewidth=0.5)
        v, i = jax.lax.top_k(l.flatten(), 1)
        plt.scatter(i, v, label='EL')
    plt.xlabel('Weights')
    plt.ylabel('Signal Strength')
    plt.legend()

    # Export
    path = result_dir_registry[dimension] + f"/training_gradient.png"
    plt.savefig(path)
    plt.close()
    pass

def checkpoint_mlp_export_plot_gradient(gradient, dimension, epoch):
    '''Visualize the MLP gradient during training and export as plot.'''
  
    # Create plot
    plt.title(f"Learning Signal {dimension}D MLP | Epoch {epoch}")
    i = 0
    for l in gradient:
        l_flat = l[0] + l[1]
        plt.plot(l_flat, linewidth=0.5) # Needs to be cleaned if MLP also uses bias trick
        v, i = jax.lax.top_k(l_flat, 1)
        plt.scatter(i, v, label=f'L{i}')
        i += 1
    plt.xlabel('Weights')
    plt.ylabel('Signal Strength')
    plt.legend()

    # Export
    path = result_dir_registry[dimension] + f"/training_gradient.png"
    plt.savefig(path)
    plt.close()
    pass

def export_plot_training_metrics (
    model_key : str,
    model_detail_str : str,
    reg : CoreRegistry,
    configs : dict,
    dimension : int):
    '''
    Create a training metrics plot, showing the trend throughout training inlcuding:
    \n- Loss
    \n- False-Negative Rate
    \n- False-Positive Rate
    '''

    # Retrieve model specific key for results and type
    model_type = configs[model_key]['type']

    # Get training metrics from registry
    loss = reg.get(model_key + core_keys['train_val_loss_key'])
    fn = reg.get(model_key + core_keys['train_fn_key'])
    fp = reg.get(model_key + core_keys['train_fp_key'])
    epochs = reg.get(model_key + core_keys['train_epoch_key'])

    # Create plot 
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    fig.suptitle(f"{model_key.split('_')[0]} {dimension}D training")
    ax.plot(epochs, loss, label='Loss')
    ax.plot(epochs, fn, label='False Negatives')
    ax.plot(epochs, fp, label='False Positives')
    ax.set_ylim(0.0, 1.0)
    fig.text(0.0, 0.005, model_detail_str, fontsize=9)

    # Get MoE additional training metrics 
    if model_type == 'moe':
        confidence = reg.get(model_key + core_keys['train_confidence_key'])
        ax.plot(epochs, confidence, label='Gate Confidence')
    elif model_type == 'mlp':
        pass
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Export plot
    path = result_dir_registry[dimension] + f"/{model_key}_{dimension}D_training.png"
    plt.legend()
    plt.tight_layout(rect=[0, 0.25, 1, 0.99])
    plt.savefig(path)
    plt.close()

def export_plot_speed_accuracy_comparison (
    reg : CoreRegistry,
    configs : dict,
    dimension : int):
    '''
    Create a speed accuray plot of all configured models.
    '''

    # Get general configs
    batch_size = configs['general']['batch_size']

    # Plot each model with its inference speed and accuracy
    for i in configs.items():
        model_key = i[0]
        if model_key == 'general': continue
        model_type = configs[model_key]['type']

        if model_type == 'moe':

            # Get model configs details 
            nex = configs[model_key]['nex']
            gate_hid_lay = configs[model_key]['gate_hidden_layer']
            expert_hid_lay = configs[model_key]['expert_hidden_layer']
            gate_arch = [dimension] + gate_hid_lay + [nex]
            expert_arch = [dimension] + expert_hid_lay + [1]
            total_p = count_parameter(expert_arch) * nex + count_parameter(gate_arch)
            active_p = count_parameter(expert_arch) + count_parameter(gate_arch)

            # Get sparse and dense results
            dkey = f'{model_key}_dense'
            skey = f'{model_key}_sparse'
            sparse_speed = reg.get(skey + core_keys['inf_speed_key'])
            sparse_fp = reg.get(skey + core_keys['fp_key'])
            dense_speed = reg.get(dkey + core_keys['inf_speed_key'])
            dense_fp = reg.get(dkey + core_keys['fp_key'])

            # Plot
            plt.plot([sparse_speed, dense_speed],[sparse_fp, dense_fp], linestyle = '--', label= f"{model_key} | ●-D | ■-S | Gate: {gate_arch} | {nex}x Experts: {expert_arch} | Active/Total P: {active_p}/{total_p}", zorder=1)
            plt.scatter(sparse_speed, sparse_fp, color='black', marker='s', zorder=2)
            plt.scatter(dense_speed, dense_fp, color='black', marker='o', zorder=2)

        elif model_type == 'mlp':
            # Get model configs details
            hid_lay = configs[model_key]['hidden_layer']
            mlp_arch = [dimension] + hid_lay + [1]
            total_p = count_parameter(mlp_arch)

            # Get results and plot
            speed = reg.get(model_key + core_keys['inf_speed_key'])
            fp = reg.get(model_key + core_keys['fp_key'])
            plt.scatter(speed, fp, label=f"{model_key}: {mlp_arch} | Active/Total P: {total_p}")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # General plot 
    plt.title(f'{dimension}D Model Comparison\non {jax.devices()[0].device_kind}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=8)
    plt.ylabel('False-Positive Rate')
    plt.xlabel(f'Inference Speed (ms) per batch ({batch_size})')
    plt.tight_layout()

    # Export plot
    path = result_dir_registry[dimension] + f"/{dimension}D_models_comparison.png"
    plt.savefig(path)
    plt.close()

def create_model_details_string (
    model_type : str,
    model_key : str,
    reg : CoreRegistry,
    configs : dict,
    dimension : int) -> str :
    '''
    Create a a string that captures important model details such as:
    \n- Architecture
    \n- Epochs
    \n- Batchsize
    \n- Parameter count
    \n- Loss function
    '''
    # Get model configurations
    epochs = reg.get(model_key + core_keys['total_epochs']),
    batch_size = configs['general']['batch_size']

    if model_type == 'moe':
        # Get model configurations
        nex = configs[model_key]['nex']
        moe_gate_arch = [dimension] + configs[model_key]['gate_hidden_layer'] + [nex]
        moe_expert_arch = [dimension] + configs[model_key]['expert_hidden_layer'] + [1]

        # Get results
        dkey = f'{model_key}_dense'
        confidence = reg.get(model_key + core_keys['gating_confidence_key'])
        total_p = reg.get(dkey + core_keys['total_parameters_key'])

        # Create string
        a = f"MoE with {nex} Experts: {moe_expert_arch}"
        b = f"\nGate: {moe_gate_arch}"
        c = f"\nTotal Parameters:{total_p}"
        d = f"\nEpochs: {epochs}, Batchsize: {batch_size}"
        e = f"\nLoss: BCE + KL + AE"
        f = f"\n\nTop1 activation mean: {round(float(confidence),2)}"
        return a + b + c + d + e + f
    
    elif model_type == 'mlp':
        # Get model configurations
        mlp_arch = [dimension] + configs[model_key]['hidden_layer'] + [1]

        # Get results
        total_p = reg.get(model_key + core_keys['total_parameters_key'])

        # Create string
        a = f"MLP: {mlp_arch}"
        b = f"\nTotal Parameters:{total_p}"
        c = f"\nEpochs: {epochs}, Batchsize: {batch_size}"
        d = f"\nLoss: BCE"

        return a + b + c + d

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
