import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mplt

from beh.registry import *
from beh.styler.registry import *
from beh.core.registry import *
from beh.core.moe import *
from beh.core.params import *
from beh.styler.shared import create_model_details_string

#------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------

def export_plot_2D_moe_internal (
    model_key : str,
    y : jax.Array,
    reg : CoreRegistry,
    configs : dict,
    dimension : int,
    threshold : float,
    model_detail_str : str,
    id : str = ''):
    '''
    2D
    \nCreate MoE internal state overview plot independtly of number of experts inlcuding:
    \n- Topk activtion map
    \n- Dense & Sparse output
    \n- TopK activation mean distribution
    \n- MoE architecture details 
    '''

    # Retrieve model specific key for results
    dkey = f'{model_key}_dense'
    skey = f'{model_key}_sparse'

    # Dimension
    img_dim_0, img_dim_1, _ = reg.get(core_keys['data_size_key'])

    # Get model configuration
    nex = configs[model_key]['nex']
    topk = 1

    # Get results from registry
    dense_yp_NOTremapped = reg.get(dkey + core_keys['y_prediciton_RAW_key'])
    sparse_yp_NOTremapped = reg.get(skey + core_keys['y_prediciton_RAW_key'])

    dense_yp = reg.get(dkey + core_keys['y_prediciton_key'])
    sparse_yp = reg.get(skey + core_keys['y_prediciton_key'])

    dense_fp = reg.get(dkey + core_keys['fp_key'])
    sparse_fp = reg.get(skey + core_keys['fp_key'])
    sparse_fn = reg.get(skey + core_keys['fn_key'])

    top1_activation = reg.get(model_key + core_keys['gate_top1_activation_key'])

    # Color prediction based on top-1 expert colors
    dense_yp_col = color_by_expert(nex, dense_yp, top1_activation)
    sparse_yp_col = color_by_expert(nex, sparse_yp, top1_activation)

    background_col = mplt.colors.to_rgba(white_gray)
    dense_mask = np.expand_dims(((dense_yp > threshold) * ( y == 0)), axis=1)
    dense_yp_col = color_by_expert(nex, dense_mask.flatten().astype(jnp.float32), top1_activation)
    dense_yp_col_fp = dense_yp_col * dense_mask + ~dense_mask * background_col

    sparse_mask = np.expand_dims(((sparse_yp > threshold) * ( y == 0)), axis=1)
    sparse_yp_col_fp = color_by_expert(nex, sparse_mask.flatten().astype(jnp.float32), top1_activation)
    sparse_yp_col_fp = sparse_yp_col * sparse_mask + ~sparse_mask * background_col

    # FN
    sparse_mask = np.expand_dims(((sparse_yp <= threshold) * ( y == 1)), axis=1)
    sparse_yp_col_fn = color_by_expert(nex, sparse_mask.flatten().astype(jnp.float32), top1_activation)
    sparse_yp_col_fn = sparse_yp_col_fn * sparse_mask + ~sparse_mask * background_col
    
    # Create plot 
    r, c = 2, 4
    fig, ax = plt.subplots(r,c, figsize=(16,9.2))
    
    ## Original
    ax[0,0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,0].set_title("Original", fontsize=9)

    ## Activation map
    ax[0,1].imshow(top1_activation.reshape((img_dim_0,img_dim_1)), vmin=0, cmap=expert_cmap, vmax=nex-1)
    ax[0,1].set_title(f"Activation Map Top-{topk} Expert", fontsize=9)

    ## Predicitions
    ax[0,2].imshow(dense_yp_col.reshape((img_dim_0,img_dim_1, -1)))
    ax[0,2].set_title(f"Model output with\n{round(float(jnp.min(dense_yp_NOTremapped)),2)} - {round(float(jnp.max(dense_yp_NOTremapped)),2)} remapped to {round(float(jnp.min(dense_yp)),2)} - {round(float(jnp.max(dense_yp)),2)}", fontsize=9)
    
    ax[0,3].imshow(sparse_yp_col.reshape((img_dim_0,img_dim_1, -1)))
    ax[0,3].set_title(f"Model output with\n{round(float(jnp.min(sparse_yp_NOTremapped)),2)} - {round(float(jnp.max(sparse_yp_NOTremapped)),2)} remapped to {round(float(jnp.min(sparse_yp)),2)} - {round(float(jnp.max(sparse_yp)),2)}", fontsize=9)

    ## FN
    ax[1,1].imshow((sparse_yp_col_fn).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,1].set_title(f"Sparse FN {round(float(sparse_fn),8)}", fontsize=9)    

    ## Conservativness
    ax[1,2].imshow((dense_yp_col_fp).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,2].set_title(f"Dense FP {round(float(dense_fp),8)}", fontsize=9)

    ax[1,3].imshow((sparse_yp_col_fp).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,3].set_title(f"Sparse FP {round(float(sparse_fp),8)}", fontsize=9)

    for i in range(r):
        for j in range(c):
            ax[i,j].axis('off')

    fig.text(0.01, 0.05, model_detail_str, fontsize=9)
    plt.tight_layout()

    # Export plot
    path = result_dir_registry[dimension] + f"/{model_key}{id}_2D_internal.png"
    plt.savefig(path)
    plt.close()

    # Export binary classification images
    ## Dense
    binary_yp = (dense_yp > threshold).reshape((img_dim_0,img_dim_1))
    yp = (binary_yp * 255).astype(np.uint8)
    plt.imsave(result_dir_registry[dimension] + f"/{model_key}{id}_dense_yp.png", yp, cmap="binary")
    ## Sparse
    binary_yp = (sparse_yp > threshold).reshape((img_dim_0,img_dim_1))
    yp = (binary_yp * 255).astype(np.uint8)
    plt.imsave(result_dir_registry[dimension] + f"/{model_key}{id}_sparse_yp.png", yp, cmap="binary")

#------------------------------------------------------------------------------------

def nex_gradients(nex: int):
    expert_gradients = []
    cmap = mplt.cm.get_cmap(expert_cmap)
    for i in jnp.linspace(0, 1, nex):
        expert_gradients.append(
            mplt.colors.LinearSegmentedColormap.from_list("mono_custom", [white_gray, cmap(i) ])
        )
    return expert_gradients

#------------------------------------------------------------------------------------

def color_by_expert(nex : int, yp : jax.Array, top1_activation : jax.Array):
    # Extract colors for all experts
    expert_gradients = nex_gradients(nex)

    # Color the image predicions based on the top1 expert color
    yp = np.array(yp)
    colored_yp = np.zeros((yp.shape[0], 4))
    for i in range(nex):
        mask = (top1_activation == i).flatten() # mask
        expert = expert_gradients[i](yp[mask]) # color based yp
        colored_yp[mask] = expert
    return colored_yp

#------------------------------------------------------------------------------------

def export_plot_2D_mlp_internal (
    model_key : str,
    y : jax.Array,
    reg : CoreRegistry,
    dimension : int,
    threshold : float,
    model_detail_str : str,
    id : str = '',):
    '''
    2D
    \nCreate MLP internal state overview plot inlcuding:
    \n- Decision boundaries
    \n- MLP architecture details 
    '''

    # Dimension
    img_dim_0, img_dim_1, _ = reg.get(core_keys['data_size_key'])
    
    # Get results from registry
    yp = reg.get(model_key + core_keys['y_prediciton_key'])
    rounded_mse = round(float(reg.get(model_key + core_keys['accuracy_mse_key'])),4)
    fp = reg.get(model_key + core_keys['fp_key'])
    
    # Create plot 
    r, c = 1, 3
    fig, ax = plt.subplots(r,c, figsize=(13,5.1))
    ax[0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0].set_title("Original", fontsize=9)
    ax[1].imshow(yp.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[1].set_title(f"Model Output", fontsize=9)
    
    # Conservativness
    binary_yp = ((yp > threshold) * ( y == 0)).reshape((img_dim_0,img_dim_1))
    ax[2].imshow(binary_yp, cmap= wb_gradient)
    ax[2].set_title(f"False Positives {round(float(fp),8)}", fontsize=9)

    for i in range(c):
        ax[i].axis('off')

    fig.text(0.013, 0.02, model_detail_str, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Export plot
    path = result_dir_registry[dimension] + f"/{model_key}{id}_2D_internal.png"
    plt.savefig(path)
    plt.close()

    # Export binary classification image
    binary_yp = (yp > threshold).reshape((img_dim_0,img_dim_1))
    yp = (binary_yp * 255).astype(np.uint8)
    plt.imsave(result_dir_registry[dimension] + f"/{model_key}{id}_yp.png", yp, cmap="binary")

#------------------------------------------------------------------------------------

def export_plot_2D_internal_comparison (
    model_key : str,
    y : jax.Array,
    reg : CoreRegistry,
    configs : dict,
    dimension : int,
    threshold : float,
    mask_experts : bool = True):
    '''
    2D
    \nCreate internal state overview plot comparison between models. 
    \nRequire configuration to be in order such that first MLP and then MoE.
    '''
    # Return if not all configured models of same size have been benchmarked yet
    config_list = list(configs.keys())
    config_model_idx = config_list.index(model_key)
    previous_model_key = config_list[config_model_idx-1]
    current_size = model_key[3:]
    # Skip if first model to evaluate or 
    if previous_model_key == 'general' or previous_model_key[3:] != current_size: return

    # Retrieve model specific key for results
    dkey = f'{model_key}_dense'
    skey = f'{model_key}_sparse'

    # Dimension
    img_dim_0, img_dim_1, _ = reg.get(core_keys['data_size_key'])

    # MOE
    # Get model configuration
    nex = configs[model_key]['nex']
    topk = 1

    # Get results from registry
    dense_yp_NOTremapped = reg.get(dkey + core_keys['y_prediciton_RAW_key'])
    sparse_yp_NOTremapped = reg.get(skey + core_keys['y_prediciton_RAW_key'])

    dense_yp = reg.get(dkey + core_keys['y_prediciton_key'])
    sparse_yp = reg.get(skey + core_keys['y_prediciton_key'])

    rounded_dense_mse = round(float(reg.get(dkey + core_keys['accuracy_mse_key'])),4)
    rounded_sparse_mse = round(float(reg.get(skey + core_keys['accuracy_mse_key'])),4)

    dense_fp = reg.get(dkey + core_keys['fp_key'])
    sparse_fp = reg.get(skey + core_keys['fp_key'])

    top1_activation = reg.get(model_key + core_keys['gate_top1_activation_key'])

    # Color prediction based on top-1 expert colors
    dense_yp_col = color_by_expert(nex, dense_yp, top1_activation)
    sparse_yp_col = color_by_expert(nex, sparse_yp, top1_activation)

    background_col = mplt.colors.to_rgba(white_gray)

    dense_mask = np.expand_dims(((dense_yp > threshold) * ( y == 0)), axis=1)
    dense_yp_fp_col = color_by_expert(nex, dense_mask.flatten().astype(jnp.float32), top1_activation)
    dense_yp_fp_col = dense_yp_fp_col * dense_mask + ~dense_mask * background_col

    sparse_mask = np.expand_dims(((sparse_yp > threshold) * ( y == 0)), axis=1)
    sparse_yp_fp_col = color_by_expert(nex, sparse_mask.flatten().astype(jnp.float32), top1_activation)
    sparse_yp_fp_col = sparse_yp_fp_col * sparse_mask + ~sparse_mask * background_col

    # MLP
    mlp_yp = reg.get(previous_model_key + core_keys['y_prediciton_key'])
    mlp_fp = reg.get(previous_model_key + core_keys['fp_key'])
    
    # Conservativness
    mlp_yp_fp = (mlp_yp > threshold) * ( y == 0)
    
    # Create combined model detail string
    moe_string = create_model_details_string(configs[model_key]['type'], model_key, reg, configs, dimension)
    mlp_string = create_model_details_string(configs[previous_model_key]['type'], previous_model_key, reg, configs, dimension)
    model_detail_str = moe_string + '\n\n' + mlp_string

    # Create plot 
    r, c = 2, 4
    fig, ax = plt.subplots(r,c, figsize=(16,9.2))
    
    ## Original
    ax[0,0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,0].set_title("Original", fontsize=9)

    ## Predicitions
    ax[0,1].imshow(mlp_yp.reshape((img_dim_0,img_dim_1, -1)), cmap= wb_gradient)
    ax[0,1].set_title(f"MLP", fontsize=9)

    ax[0,2].imshow(dense_yp_col.reshape((img_dim_0,img_dim_1, -1)))
    ax[0,2].set_title(f"Dense MoE", fontsize=9)
    
    ax[0,3].imshow(sparse_yp_col.reshape((img_dim_0,img_dim_1, -1)))
    ax[0,3].set_title(f"Sparse MoE", fontsize=9)
    
    ## Conservativness
    ax[1,1].imshow((mlp_yp_fp).reshape((img_dim_0,img_dim_1, -1)), cmap= wb_gradient)
    ax[1,1].set_title(f"FP {float(mlp_fp):.4f}", fontsize=9)

    ax[1,2].imshow((dense_yp_fp_col).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,2].set_title(f"FP {float(dense_fp):.4f}", fontsize=9)

    ax[1,3].imshow((sparse_yp_fp_col).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,3].set_title(f"FP {float(sparse_fp):.4f}", fontsize=9)

    for i in range(r):
        for j in range(c):
            ax[i,j].axis('off')

    fig.text(0.01, 0.05, model_detail_str, fontsize=9)
    plt.tight_layout()
    # Export plot
    path = result_dir_registry[dimension] + f"/{current_size}_2D_internal_comparison.png"
    plt.savefig(path)
    plt.close()