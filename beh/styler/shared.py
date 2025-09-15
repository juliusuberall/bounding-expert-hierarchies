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
        active_experts = reg.get(model_key + core_keys['active_experts_key'])
        ax.plot(epochs, active_experts, label='Active Experts')
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
        e = f"\nTop1 activation mean: {round(float(confidence),2)}"
        return a + b + c + d + e
    
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

def export_plot_model_comparison_graph (
    reg : CoreRegistry,
    configs : dict,
    dimension : int):
    '''
    Creates performance plot which shows FP rate in relation to inference speed.
    '''
    # Get results
    ## MLP
    mlps_speed, mlps_fp = [], []
    for c in configs.items():
        model_key = c[0]
        if model_key[:3] == 'mlp':
            mlps_fp.append(reg.get(model_key + core_keys['fp_key']))
            mlps_speed.append(reg.get(model_key + core_keys['inf_speed_key'])[0][1])
    mlps_fp = np.array(mlps_fp)
    mlps_speed = np.array(mlps_speed)

    ## Sparse MoE
    moes_speed, moes_fp = [], []
    for c in configs.items():
        model_key = c[0]
        if model_key[:3] == 'moe':
            skey = f'{model_key}_sparse'
            moes_fp.append(reg.get(skey + core_keys['fp_key']))
            moes_speed.append(reg.get(skey + core_keys['inf_speed_key'])[0][1])
    moes_fp = np.array(moes_fp)
    moes_speed = np.array(moes_speed)

    # Compute max and min axis values to ensure square plot
    padding = 0.05
    all_x = np.concatenate((mlps_speed, moes_speed))
    ## normalize to ensure proper plotting
    mlps_speed /= all_x.max()
    moes_speed /= all_x.max()
    all_x /= all_x.max()
    all_y = np.concatenate((mlps_fp, moes_fp))
    ## normalize to ensure proper plotting
    mlps_fp /= all_y.max()
    moes_fp /= all_y.max()
    all_y /= all_y.max()
    min_lim = min(all_x.min(), all_y.min()) - padding
    max_lim = max(all_x.max(), all_y.max()) + padding

    # Plot
    fig, ax = plt.subplots()

    # Force plot to be square
    ax.set_aspect('equal')
    lims = [min_lim, max_lim]
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.plot(mlps_speed, mlps_fp, "-o", color=cols['mlp'])
    plt.plot(moes_speed, moes_fp, "-o", color=cols['moe'])

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove outline
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set background color
    ax.set_facecolor(cols['light_gray'])

    # Get limits
    pad = 0.03
    fsize = 10
    fsize_l = 14
    # Add custom text labels along axes
    ax.text(min_lim + pad, min_lim - pad, "Fast", va="top", ha="left", fontsize=fsize, color=cols['dark_gray'], family="serif")   # bottom-left
    ax.text(max_lim - pad, min_lim - pad, "Slow", va="top", ha="right", fontsize=fsize, color=cols['dark_gray'], family="serif")  # bottom-right
    ax.text(min_lim + (max_lim-min_lim) / 2 + pad*2, min_lim - pad, "Speed", va="top", ha="right", fontsize=fsize_l, family="serif")  # bottom-center
    ax.text(min_lim - pad * 2, min_lim + pad, "Good", va="bottom", ha="left", rotation=90, fontsize=fsize, color=cols['dark_gray'], family="serif")  # left-bottom
    ax.text(min_lim - pad * 2, max_lim - pad, "Bad", va="top", ha="left", rotation=90, fontsize=fsize, color=cols['dark_gray'], family="serif")     # left-top
    ax.text(min_lim - pad * 2, min_lim + (max_lim-min_lim)/ 2 + pad *1.8, "FP", va="top", ha="left", rotation=90, fontsize=fsize_l, family="serif")     # left-center

    # Export as PDF
    fig.set_size_inches(5, 5)
    path = result_dir_registry[dimension] + f"/{dimension}D_model_comparison_graph.pdf"
    plt.savefig(path , bbox_inches=None, pad_inches=0)