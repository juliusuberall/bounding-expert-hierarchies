import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mplt
import cv2

from beh.registry import *
from beh.styler.registry import *
from beh.core.registry import *
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

def export_plot_2D_moe_grid_internal (
    data_name : str, 
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
    \nCreate grid based MoE internal state overview plot independtly of number of experts inlcuding:
    \n- Topk activtion map
    \n- MoE architecture details 
    '''

    # Dimension
    img_dim_0, img_dim_1, _ = reg.get(core_keys['data_size_key'])

    # Get model configuration
    grid_dim = configs[model_key]['grid_dim']
    nex = grid_dim ** dimension
    topk = 1

    # Get results from registry
    yp_raw = reg.get(model_key + core_keys['y_prediciton_RAW_key'])
    yp = reg.get(model_key + core_keys['y_prediciton_key'])
    fp = reg.get(model_key + core_keys['fp_key'])
    fn = reg.get(model_key + core_keys['fn_key'])
    idx = reg.get(model_key + core_keys['gate_top1_activation_key'])

    # Color prediction based on top-1 expert colors
    yp_col = color_by_expert(nex, yp, idx)

    background_col = mplt.colors.to_rgba(back_col)
    mask = np.expand_dims(((yp > threshold) * ( y == 0)), axis=1)
    yp_col_fp = color_by_expert(nex, mask.flatten().astype(jnp.float32), idx)
    yp_col_fp = yp_col * mask + ~mask * background_col

    # FN
    mask = np.expand_dims(((yp <= threshold) * ( y == 1)), axis=1)
    yp_col_fn = color_by_expert(nex, mask.flatten().astype(jnp.float32), idx)
    yp_col_fn = yp_col_fn * mask + ~mask * background_col
    
    # Create plot 
    r, c = 2, 3
    fig, ax = plt.subplots(r,c, figsize=(16,9.2))
    
    ## Original
    ax[0,0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0,0].set_title("Original", fontsize=9)

    ## Activation map
    ax[0,1].imshow(idx.reshape((img_dim_0,img_dim_1)), vmin=0, cmap=expert_cmap, vmax=nex-1)
    ax[0,1].set_title(f"Expert Grid", fontsize=9)

    ## Predicitions
    ax[0,2].imshow(yp_col.reshape((img_dim_0,img_dim_1, -1)))
    ax[0,2].set_title(f"Model output with\n{round(float(jnp.min(yp_raw)),2)} - {round(float(jnp.max(yp_raw)),2)} remapped to {round(float(jnp.min(yp)),2)} - {round(float(jnp.max(yp)),2)}", fontsize=9)
    
    ## FN
    ax[1,1].imshow((yp_col_fn).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,1].set_title(f"FN {round(float(fn),8)}", fontsize=9)    

    ## Conservativness
    ax[1,2].imshow((yp_col_fp).reshape((img_dim_0,img_dim_1, -1)))
    ax[1,2].set_title(f"FP {round(float(fp),8)}", fontsize=9)

    for i in range(r):
        for j in range(c):
            ax[i,j].axis('off')

    fig.text(0.01, 0.05, model_detail_str, fontsize=9)
    plt.tight_layout()

    # Export plot
    path = result_dir_registry[dimension] + f"/{data_name}_{model_key}{id}_2D_internal.png"
    plt.savefig(path)
    plt.close()

#------------------------------------------------------------------------------------

def export_plot_2D_moe_internal (
    data_name : str, 
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

    background_col = mplt.colors.to_rgba(back_col)
    dense_mask = np.expand_dims(((dense_yp > threshold) * ( y == 0)), axis=1)
    dense_yp_col_fp = color_by_expert(nex, dense_mask.flatten().astype(jnp.float32), top1_activation)
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
    path = result_dir_registry[dimension] + f"/{data_name}_{model_key}{id}_2D_internal.png"
    plt.savefig(path)
    plt.close()

#------------------------------------------------------------------------------------

def nex_gradients(nex: int):
    expert_gradients = []
    cmap = mplt.cm.get_cmap(expert_cmap)
    for i in jnp.linspace(0, 1, nex):
        expert_gradients.append(
            mplt.colors.LinearSegmentedColormap.from_list("mono_custom", [back_col, cmap(i) ])
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
    data_name : str,
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
    fp = reg.get(model_key + core_keys['fp_key'])
    
    # Create plot 
    r, c = 1, 3
    fig, ax = plt.subplots(r,c, figsize=(13,5.1))
    ax[0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0].set_title("Original", fontsize=9)
    ax[1].imshow(yp.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[1].set_title(f"Model output with\n{round(float(jnp.min(yp)),2)} - {round(float(jnp.max(yp)),2)} ", fontsize=9)
    
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

#------------------------------------------------------------------------------------

def export_plot_2D_internal_comparison (
    model_key : str,
    y : jax.Array,
    reg : CoreRegistry,
    configs : dict,
    dimension : int,
    threshold : float):
    '''
    2D
    \nCreate internal state overview plot comparison between models. Shows raw model output which might be helpful for debugging.
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

    # Get results from registry
    dense_yp = reg.get(dkey + core_keys['y_prediciton_key'])
    sparse_yp = reg.get(skey + core_keys['y_prediciton_key'])

    dense_fp = reg.get(dkey + core_keys['fp_key'])
    sparse_fp = reg.get(skey + core_keys['fp_key'])

    top1_activation = reg.get(model_key + core_keys['gate_top1_activation_key'])

    # Color prediction based on top-1 expert colors
    dense_yp_col = color_by_expert(nex, dense_yp, top1_activation)
    sparse_yp_col = color_by_expert(nex, sparse_yp, top1_activation)

    background_col = mplt.colors.to_rgba(back_col)

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

#------------------------------------------------------------------------------------

def export_plot_2D_binary_comparison_paper_row (
    data_name : str,
    model_key : str,
    y : jax.Array,
    reg : CoreRegistry,
    configs : dict,
    dimension : int,
    threshold : float,
    export_binary : bool = False):
    '''
    2D
    \nCreate a binary classification comparison plot. A row with scene, mlp, dense and sparse moe. 
    \nIntended for paper figure.
    \nRequires configuration to be in order such that first MLP and then MoE.
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
    s = configs['general']['aa_scaling'] 

    # MOE
    ## Get model configuration
    nex = configs[model_key]['nex']

    ## Query MoE in double resolution and anti-alias 
    dense_yp = reg.get(dkey + core_keys['aa_y_prediciton_key'])
    sparse_yp = reg.get(skey + core_keys['aa_y_prediciton_key'])

    ## Color densen and sparse classification based on top1 experts
    back_col = (1.0, 1.0, 1.0, 0.0) # background is transparent, make bright white to avoid outlines
    back = (0.0, 0.0, 0.0, 0.0) # background is transparent, make bright white to avoid outlines
    top1_activation = reg.get(model_key + core_keys['aa_gate_top1_activation_key'])

    dense_mask = np.expand_dims(dense_yp > threshold, axis=1)
    dense_binary_col = color_by_expert(nex, dense_mask.flatten().astype(jnp.float32), top1_activation)
    dense_binary_col = dense_binary_col * dense_mask + ~dense_mask * back_col
    dense_binary_col = dense_binary_col.reshape((img_dim_0*s, img_dim_1*s, -1))

    sparse_mask = np.expand_dims(sparse_yp > threshold, axis=1)
    sparse_binary_col = color_by_expert(nex, sparse_mask.flatten().astype(jnp.float32), top1_activation)
    sparse_binary_col = sparse_binary_col * sparse_mask + ~sparse_mask * back_col
    sparse_binary_col = sparse_binary_col.reshape((img_dim_0*s, img_dim_1*s, -1))

    # MLP
    mlp_yp = reg.get(previous_model_key + core_keys['aa_y_prediciton_key'])
    mask = np.expand_dims(mlp_yp > threshold, axis=1)
    mlp_binary = (0.0, 0.0, 0.0, 1.0) * mask + ~mask * back
    mlp_binary = mlp_binary.reshape((img_dim_0*s, img_dim_1*s, -1))

    # Anti-Alias
    interpolation = cv2.INTER_AREA
    dense_binary_col = cv2.resize(dense_binary_col, None, fx=1/s, fy=1/s, interpolation=interpolation)
    sparse_binary_col = cv2.resize(sparse_binary_col, None, fx=1/s, fy=1/s, interpolation=interpolation)
    mlp_binary = cv2.resize(mlp_binary, None, fx=1/s, fy=1/s, interpolation=interpolation)

    ## OpenCV swaps width and height such that we have to swap back
    dense_binary_col = np.transpose(dense_binary_col, (1, 0, 2))
    sparse_binary_col = np.transpose(sparse_binary_col, (1, 0, 2))
    mlp_binary = np.transpose(mlp_binary, (1, 0, 2))

    dense_binary_col = np.clip(dense_binary_col, 0, 1)
    sparse_binary_col = np.clip(sparse_binary_col, 0, 1)
    mlp_binary = np.clip(mlp_binary, 0, 1)
    
    # Create combined model detail string
    moe_string = create_model_details_string(configs[model_key]['type'], model_key, reg, configs, dimension)
    mlp_string = create_model_details_string(configs[previous_model_key]['type'], previous_model_key, reg, configs, dimension)
    model_detail_str = moe_string + '\n\n' + mlp_string

    # Create plot 
    r, c = 1, 4
    fig, ax = plt.subplots(r,c, figsize=(16,6))
    
    ## Original
    ax[0].imshow(y.reshape((img_dim_0,img_dim_1)), cmap= wb_gradient)
    ax[0].set_title("Original", fontsize=9)
    
    ## Conservativness
    mlp_fp = reg.get(previous_model_key + core_keys['fp_key'])
    ax[1].imshow(mlp_binary)
    ax[1].set_title(f"MLP | FP {float(mlp_fp):.4f}", fontsize=9)

    dense_fp = reg.get(dkey + core_keys['fp_key'])
    ax[2].imshow(dense_binary_col)
    ax[2].set_title(f"Dense MoE | FP {float(dense_fp):.4f}", fontsize=9)

    sparse_fp = reg.get(skey + core_keys['fp_key'])
    ax[3].imshow(sparse_binary_col)
    ax[3].set_title(f"Sparse MoE | FP {float(sparse_fp):.4f}", fontsize=9)

    for i in range(c):
        ax[i].axis('off')

    fig.text(0.01, 0.04, model_detail_str, fontsize=9)
    plt.tight_layout(rect=(0.0,0.25,1.0,1.0))
    # Export plot
    path = result_dir_registry[dimension] + f"/{data_name}_{current_size}_2D_binary_comparison.png"
    plt.savefig(path)
    plt.close()

    # Export binary classification image for mlp and sparse Moe.
    # We want to export with an alpha channel and not as previously on background.
    if not export_binary: return

    ## MLP
    plt.imsave(result_dir_registry[dimension] + f"/{data_name}_{current_size}_mlp_binary.png", mlp_binary)

    ## Dense MoE
    dense_binary = (0.0, 0.0, 0.0, 1.0) * dense_mask + ~dense_mask * back
    dense_binary = dense_binary.reshape((img_dim_0* s,img_dim_1 * s, -1))
    dense_binary = cv2.resize(dense_binary, None, fx=1/s, fy=1/s, interpolation=interpolation)
    dense_binary = np.transpose(dense_binary, (1, 0, 2))
    dense_binary = np.clip(dense_binary, 0, 1)
    plt.imsave(result_dir_registry[dimension] + f"/{data_name}_{current_size}_denseMoE_binary.png", dense_binary)

    ## Sparse MoE
    sparse_binary = (0.0, 0.0, 0.0, 1.0) * sparse_mask + ~sparse_mask * back
    sparse_binary = sparse_binary.reshape((img_dim_0 * s,img_dim_1 * s, -1))
    sparse_binary = cv2.resize(sparse_binary, None, fx=1/s, fy=1/s, interpolation=interpolation)
    sparse_binary = np.transpose(sparse_binary, (1, 0, 2))
    sparse_binary = np.clip(sparse_binary, 0, 1)
    plt.imsave(result_dir_registry[dimension] + f"/{data_name}_{current_size}_sparseMoE_binary.png", sparse_binary)

    # Sparse MoE expert color
    plt.imsave(result_dir_registry[dimension] + f"/{data_name}_{current_size}_sparseMoE_binary_col.png", sparse_binary_col)