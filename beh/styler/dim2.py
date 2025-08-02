import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mplt

from datetime import datetime

from beh.registry import *
from beh.styler.registry import *
from beh.core.registry import *
from beh.core.moe import *
from beh.core.params import *

def export_plot_training_data (x : jax.Array, y : jax.Array):
    '''
    2D
    \nExport image of the alpha channel extracted as labels of the training data.
    '''
    
    # Retrieve image width and height from data
    xdim = jnp.unique(x[...,0]).size
    ydim = jnp.unique(x[...,1]).size

    # Create image using label data and export
    img = y.reshape((xdim,ydim))
    dimension = 2
    path = result_dir_registry[dimension] + f"/y.png"
    plt.imsave(
        path,
        img,
        cmap = cmap
    )
    print(f"2D training data plot saved at {path}")
    pass

def export_plot_2D_moe_internal_8_experts (
    model_key : str,
    x : jax.Array,
    y : jax.Array,
    reg : CoreRegistry,
    config : dict,
    dimension : int,
    threshold : float,
    mask_experts : bool = True,):
    '''
    2D
    \nCreate MoE internal state overview plot inlcuding:
    \n- Expert decision boundaries
    \n- Topk activtion map
    \n- Dense & Sparse output
    \n- TopK activation mean distribution
    \n- MoE architecture details 
    '''

    # Retrieve model specific key for results
    dkey = f'{model_key}_{dimension}_dense'
    skey = f'{model_key}_{dimension}_sparse'

    # Dimension
    dim = 2
    img_dim_0, img_dim_1, _ = reg.get(core_keys['data_resolution_key'])

    # Get model configuration
    nex = configs[model_key]['nex']
    epochs = configs['general']['epochs']
    batch_size = configs['general']['batch_size']
    moe_gate_arch = [dim] + configs[model_key]['gate_hidden_layer'] + [nex]
    moe_expert_arch = [dim] + configs[model_key]['expert_hidden_layer'] + [1]

    # Get results from registry
    model_key = f'{model_key}_{dimension}'
    dense_yp_NOTremapped = reg.get(dkey + core_keys['y_prediciton_RAW_key'])
    sparse_yp_NOTremapped = reg.get(skey + core_keys['y_prediciton_RAW_key'])

    dense_yp = reg.get(dkey + core_keys['y_prediciton_key'])
    sparse_yp = reg.get(skey + core_keys['y_prediciton_key'])

    rounded_dense_mse = round(float(reg.get(dkey + core_keys['accuracy_mse_key'])),4)
    rounded_sparse_mse = round(float(reg.get(skey + core_keys['accuracy_mse_key'])),4)

    sparse_fn = reg.get(skey + core_keys['sparse_fn_key'])
    sparse_fp = reg.get(skey + core_keys['sparse_fp_key'])

    top1_activation = reg.get(model_key + core_keys['gate_top1_activation_key'])
    gate_sorted_activation = reg.get(model_key + core_keys['gating_sorted_activation_key'])
    confidence = reg.get(model_key + core_keys['gating_confidence_key'])
    e_decBoundaries = reg.get(model_key + core_keys['expert_boundary_key'])
    
    ############ We might want to outsource this in a central colour system
    # Extract colors for all experts
    expert_gradients = []
    viridis = mplt.cm.get_cmap('viridis')
    for i in jnp.linspace(0,1,nex):
        expert_gradients.append(
            mplt.colors.LinearSegmentedColormap.from_list("mono_custom", ["whitesmoke", viridis(i) ])
        )
    ## Compute expert greys to avoid full white that is invisible
    expGreys = mplt.cm.get_cmap('Greys')(jnp.linspace(0,1,nex+1)[jnp.arange(nex)[::-1]])
    expGreys[-1,...] = [0.98, 0.98, 0.98,1.]
    ## Custom gray to black gradient
    wb_gradient = mplt.colors.LinearSegmentedColormap.from_list("mono_custom", ["whitesmoke", "black"])

    # Mask expert decision boundaries
    if mask_experts:
        for i in range(nex):
            e_decBoundaries[i] *= (top1_activation == i).flatten()

    # Create plot 
    r, c = 4, 4
    fig, ax = plt.subplots(r,c, figsize=(15,16))
    ax[0,0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,0].set_title("Original", fontsize=9)
    ax[0,1].imshow(dense_yp.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,1].set_title(f"Dense Predicition MSE {rounded_dense_mse} with\n{round(float(jnp.min(dense_yp_NOTremapped)),2)} - {round(float(jnp.max(dense_yp_NOTremapped)),2)} remapped to {round(float(jnp.min(dense_yp)),2)} - {round(float(jnp.max(dense_yp)),2)}", fontsize=9)
    ax[0,2].imshow(sparse_yp.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,2].set_title(f"Sparse Prediction MSE {rounded_sparse_mse} with\n{round(float(jnp.min(sparse_yp_NOTremapped)),2)} - {round(float(jnp.max(sparse_yp_NOTremapped)),2)} remapped to {round(float(jnp.min(sparse_yp)),2)} - {round(float(jnp.max(sparse_yp)),2)}", fontsize=9)
    ax[0,3].imshow(top1_activation.reshape((img_dim_0,img_dim_1)), vmin=0, vmax=nex-1)
    ax[0,3].set_title(f"Activation Map Top {topk} Expert", fontsize=9)
    ax[1,0].set_title(f"Expert 1", fontsize=9)
    ax[1,0].imshow(e_decBoundaries[0].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[0])
    ax[1,1].set_title(f"Expert 2", fontsize=9)
    ax[1,1].imshow(e_decBoundaries[1].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[1])
    ax[1,2].set_title(f"Expert 3", fontsize=9)
    ax[1,2].imshow(e_decBoundaries[2].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[2])
    ax[1,3].set_title(f"Expert 4", fontsize=9)
    ax[1,3].imshow(e_decBoundaries[3].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[3])
    ax[2,0].set_title(f"Expert 5", fontsize=9)
    ax[2,0].imshow(e_decBoundaries[4].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[4])
    ax[2,1].set_title(f"Expert 6", fontsize=9)
    ax[2,1].imshow(e_decBoundaries[5].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[5])
    ax[2,2].set_title(f"Expert 7", fontsize=9)
    ax[2,2].imshow(e_decBoundaries[6].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[6])
    ax[2,3].set_title(f"Expert 8", fontsize=9)
    ax[2,3].imshow(e_decBoundaries[7].reshape((img_dim_0,img_dim_1)), cmap=expert_gradients[7])

    for i in range(r):
        for j in range(c):
            ax[i,j].axis('off')

    # Conservativness
    ax[3,0].imshow(((sparse_yp < threshold) * ( y > 0)).reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[3,0].set_title(f"Sparse False Negatives {round(float(sparse_fn),8)}", fontsize=9)
    ax[3,1].imshow(((sparse_yp > threshold) * ( y == 0)).reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[3,1].set_title(f"Sparse False Positives {round(float(sparse_fp),8)}", fontsize=9)

    # TopK activations
    ax[3,2].set_title(f" \nExpert Activation - TopK sorting", fontsize=9)
    for i in range(nex):
        eActi = gate_sorted_activation[...,nex-1-i].flatten()
        ax[3,2].plot(jnp.arange(eActi.size), eActi, color= expGreys[i])
    ax[3,2].axis('on')
    ax[3,2].set_xticks([])  
    ax[3,2].set_xticklabels([]) 
    ax[3,2].set_xlabel("Point")
    ax[3,2].set_ylabel("Probability")

    ax[3,3].set_title(f"Activation Mean - TopK sorted", fontsize=9)
    ax[3,3].bar(jnp.arange(nex) + 1,
                jnp.mean(gate_sorted_activation.reshape((-1, nex)).T, axis=1)[::-1],
                color = expGreys
                )
    ax[3,3].axis('on')
    ax[3,3].set_xticks([]) 
    ax[3,3].set_xticklabels([]) 
    ax[3,3].set_xlabel("TopK")
    ax[3,3].set_ylabel("Probability")


    fig.text(0.78, 0.01, f"MoE with {nex} Experts: {moe_expert_arch} \nGate: {moe_gate_arch}\nTotal Parameters MoE:{count_parameter(moe_expert_arch) * nex + count_parameter(moe_gate_arch)}\nEpochs: {epochs}, Batchsize: {batch_size}\nLoss: BCE + KL + AE\n\nTop1 activation mean: {round(float(confidence),2)}", fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.suptitle(f'{model_key} Decision Boundary Overview\nLoss: BCE + KL + AE')
    
    # Export plot
    timestamp = ""
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
    path = result_dir_registry[dimension] + f"/{timestamp}moe_internal.png"
    plt.savefig(path)
    plt.close()

    pass