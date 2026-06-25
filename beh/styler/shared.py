import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beh.registry import *
from beh.core.registry import *
from beh.core.params import count_parameter
from beh.styler.registry import cols

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

#------------------------------------------------------------------------------------

# Plot the gradient of a neural network training
def checkpoint_moeg_export_plot_gradient(gradient, dimension, epoch):
    '''Visualize the MoE gradient during training and export as plot.'''
  
    # Create plot
    plt.title(f"Learning Signal {dimension}D MoE Grid | Epoch {epoch}")
    for l in gradient:
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

#------------------------------------------------------------------------------------

def checkpoint_mlp_export_plot_gradient(gradient, dimension, epoch):
    '''Visualize the MLP gradient during training and export as plot.'''
  
    # Create plot
    plt.title(f"Learning Signal {dimension}D MLP | Epoch {epoch}")
    i = 0
    for l in gradient:
        plt.plot(l.flatten(), linewidth=0.5) # Needs to be cleaned if MLP also uses bias trick
        v, i = jax.lax.top_k(l.flatten(), 1)
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

#------------------------------------------------------------------------------------

def export_plot_training_metrics (
    data_name : str,
    model_key : str,
    model_detail_str : str,
    reg : CoreRegistry,
    configs : dict,
    dimension : int):
    '''
    Create a training metrics plot, showing the trend throughout training inlcuding:
    \n- False-Negative Rate
    \n- False-Positive Rate
    \n- Gate Confidence
    \n- Active Experts
    \n- Conservative Experts
    '''

    # Retrieve model specific key for results and type
    model_type = configs[model_key]['type']

    # Get training metrics from registry
    fn = reg.get(model_key + core_keys['train_fn_key'])
    fp = reg.get(model_key + core_keys['train_fp_key'])
    epochs = reg.get(model_key + core_keys['train_epoch_key'])

    # Create plot 
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    fig.suptitle(f"{model_key.split('_')[0]} {dimension}D training")
    ax.plot(epochs, fn, label='False Negatives')
    ax.plot(epochs, fp, label='False Positives')
    ax.set_ylim(0.0, 1.0)
    fig.text(0.0, 0.005, model_detail_str, fontsize=9)

    # Get MoE additional training metrics 
    if model_type == 'moe':
        confidence = reg.get(model_key + core_keys['train_confidence_key'])
        ax.plot(epochs, confidence, label='Gate Confidence')
        active_experts = reg.get(model_key + core_keys['active_experts_key'])
        ax.plot(epochs, active_experts, label='Active Experts')
        con_experts = reg.get(model_key + core_keys['train_conservative_experts_key'])
        ax.plot(epochs, con_experts, label='Conservative Experts')
    elif model_type == 'moeg':
        con_experts = reg.get(model_key + core_keys['train_conservative_experts_key'])
        ax.plot(epochs, con_experts, label='Conservative Experts')
    elif model_type == 'mlp':
        pass
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Export plot
    path = result_dir_registry[dimension] + f"/{data_name}_{model_key}_{dimension}D_training.png"
    plt.legend()
    plt.tight_layout(rect=[0, 0.25, 1, 0.99])
    plt.savefig(path)
    plt.close()

#------------------------------------------------------------------------------------

def create_neural_model_details_string (
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
        e = f"\nTop1 activation mean: {round(float(confidence),2)}"
        return a + b + c + d + e
    
    elif model_type == 'moeg':
        # Get model configurations
        e_arch = [dimension] + configs[model_key]['expert_hidden_layer'] + [1]
        nex = configs[model_key]['grid_dim'] ** dimension

        # Get results
        total_p = reg.get(model_key + core_keys['total_parameters_key'])

        # Create string
        a = f"Grid with {nex} Experts: {e_arch}"
        b = f"\nTotal Parameters:{total_p}"
        c = f"\nEpochs: {epochs}, Batchsize: {batch_size}"

        return a + b + c
    
    elif model_type == 'mlp':
        # Get model configurations
        mlp_arch = [dimension] + configs[model_key]['hidden_layer'] + [1]

        # Get results
        total_p = reg.get(model_key + core_keys['total_parameters_key'])

        # Create string
        a = f"MLP: {mlp_arch}"
        b = f"\nTotal Parameters:{total_p}"
        c = f"\nEpochs: {epochs}, Batchsize: {batch_size}"

        return a + b + c

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
#------------------------------------------------------------------------------------
