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
    
    # Retrieve image width and height from data
    xdim = jnp.unique(x[...,0]).size
    ydim = jnp.unique(x[...,1]).size

    # Create image using label data and export
    img = y.reshape((xdim,ydim))
    dimensions = 2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = result_dir_registry[dimensions] + f"/{timestamp}_dim2_y.png"
    plt.imsave(
        path,
        img,
        cmap = cmap
    )
    print(f"2D training data plot saved at {path}")
    pass

def export_plot_2D_moe_internal_8_experts (x : jax.Array, y : jax.Array, reg : CoreRegistry, config : dict, dimensions : int):

    # Retrieve model specific key for results
    model_key = 'moe'
    dkey = model_key + '_dense_'
    skey = model_key + '_sparse_'

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
    dense_yp_NOTremapped = reg.get(dkey + core_keys['y_prediciton_RAW_key'])
    sparse_yp_NOTremapped = reg.get(skey + core_keys['y_prediciton_RAW_key'])

    dense_yp = reg.get(dkey + core_keys['y_prediciton_key'])
    sparse_yp = reg.get(skey + core_keys['y_prediciton_key'])

    rounded_dense_mse = round(float(reg.get(dkey + core_keys['accuracy_mse_key'])),4)
    rounded_sparse_mse = round(float(reg.get(skey + core_keys['accuracy_mse_key'])),4)

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

    ############

    # Print results
    wb_gradient = mplt.colors.LinearSegmentedColormap.from_list("mono_custom", ["whitesmoke", "black"])

    r, c = 4, 4
    fig, ax = plt.subplots(r,c, figsize=(15,18))
    ax[0,0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,0].set_title("Original", fontsize=9)
    ax[0,1].imshow(dense_yp.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,1].set_title(f"Dense Predicition MSE {rounded_dense_mse} with\n{round(float(jnp.min(dense_yp_NOTremapped)),2)} - {round(float(jnp.max(dense_yp_NOTremapped)),2)} remapped to {round(float(jnp.min(dense_yp)),2)} - {round(float(jnp.max(dense_yp)),2)}", fontsize=9)
    ax[0,2].imshow(sparse_yp.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,2].set_title(f"Sparse Prediction MSE {rounded_sparse_mse} with\n{round(float(jnp.min(sparse_yp_NOTremapped)),2)} - {round(float(jnp.max(sparse_yp_NOTremapped)),2)} remapped to {round(float(jnp.min(sparse_yp)),2)} - {round(float(jnp.max(sparse_yp)),2)}", fontsize=9)
    ax[0,3].imshow(top1_activation.reshape((img_dim_0,img_dim_1)))
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

    ax[3,1].set_title(f"Expert Activation all Points\nTopK sorting", fontsize=9)
    for i in range(nex):
        eActi = gate_sorted_activation[...,nex-1-i].flatten()
        ax[3,1].plot(jnp.arange(eActi.size), eActi, color= expGreys[i])
    ax[3,1].axis('on')
    ax[3,1].set_xticks([])  
    ax[3,1].set_xticklabels([]) 
    ax[3,1].set_xlabel("Point")
    ax[3,1].set_ylabel("Probability")

    ax[3,2].set_title(f"Expert Activation first 1000 Points\nTopK sorting", fontsize=9)
    for i in range(nex):
        eActi = gate_sorted_activation[...,nex-1-i].flatten()
        eActi = eActi[0:1000]
        ax[3,2].plot(jnp.arange(eActi.size), eActi, color= expGreys[i])
    ax[3,2].axis('on')
    ax[3,2].set_xticks([])  
    ax[3,2].set_xticklabels([]) 
    ax[3,2].set_xlabel("Point")
    ax[3,2].set_ylabel("Probability")

    ax[3,3].set_title(f"Activation Mean\nTopK sorted", fontsize=9)
    ax[3,3].bar(jnp.arange(nex) + 1,
                jnp.mean(gate_sorted_activation.reshape((-1, nex)).T, axis=1)[::-1],
                color = expGreys
                )
    ax[3,3].axis('on')
    ax[3,3].set_xticks([]) 
    ax[3,3].set_xticklabels([]) 
    ax[3,3].set_xlabel("TopK")
    ax[3,3].set_ylabel("Probability")


    ax[3,0].set_visible(False)

    fig.text(0.02, 0.021, f"MoE with {nex} Experts: {moe_expert_arch} \nGate: {moe_gate_arch}\nTotal Parameters MoE:{count_parameter(moe_expert_arch) * nex + count_parameter(moe_gate_arch)}\nEpochs: {epochs}, Batchsize: {batch_size}\nImportance auxiliary loss for uniform gating.\nThe MoE is queried sparse with top {topk} experts.\n\nTop1 activation mean: {confidence}", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f'{model_key} Decision Boundary Overview\nLoss: BCE + KL + AE')
    
    # Export plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = result_dir_registry[dimensions] + f"/{timestamp}_dim2_moe_internal_.png"
    plt.savefig(path)
    plt.close()
    print(f"\n2D MoE internal state plot saved at {path}")

    pass