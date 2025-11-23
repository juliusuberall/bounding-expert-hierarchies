import jax

from beh.core.registry import *
from beh.styler.shared import export_plot_training_metrics, create_model_details_string
from beh.styler.dim2 import export_plot_training_data, export_plot_2D_mlp_internal, export_plot_2D_moe_internal, export_plot_2D_moe_grid_internal, export_plot_2D_internal_comparison, export_plot_2D_binary_comparison_paper_row
from beh.styler.dim3 import prep_openVDB, marching_cube
from beh.styler.dim4 import prep_openVDB_frames
from beh.styler.dim4plus import pose_marching_cube
from beh.gsheets_registry import gsheet_log_results

def checkpoint_plot_training_data(x : jax.Array, y : jax.Array, dimension : int ):
    '''
    Validation visualization of training data.
    \nDelegates to corresponding data dimensionality sub-routine.
    '''
    if dimension == 2:
        export_plot_training_data(x, y)
        pass
    elif dimension == 3:
        pass
    elif dimension == 4:
        pass
    elif dimension == 9:
        pass
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")

def format_export_results(
    model,
    model_key : str,
    y : jax.Array, 
    reg : CoreRegistry, 
    configs : dict,
    dimension : int,
    data_name : str):
    '''
    Create result plots.
    \nDelegates to corresponding data dimensionality sub-routine.
    '''
    # Get general configs
    threshold = configs['general']['boundary_threshold']
    model_type = configs[model_key]['type']

    # Create model detail string
    model_detail_str = create_model_details_string(model_type, model_key, reg, configs, dimension) 

    # Result plots relevant for all dimensions and models
    export_plot_training_metrics(data_name, model_key, model_detail_str, reg, configs, dimension)

    # Create model internal state overview
    if model_type == 'moe':
        if dimension == 2:
            export_plot_2D_moe_internal(data_name, model_key, y, reg, configs, dimension, threshold, model_detail_str)
            #export_plot_2D_internal_comparison(model_key, y, reg, configs, dimension, threshold)
            export_plot_2D_binary_comparison_paper_row(data_name, model_key, y, reg, configs, dimension, threshold, export_binary=True)
        elif dimension == 3:
            prep_openVDB(data_name, dimension, 100, model, model_key, configs, reg)
            marching_cube(data_name, dimension, 100, model, model_key, configs, reg)
        elif dimension == 4:
            prep_openVDB_frames(data_name, dimension, 100, model, model_key, configs, reg)
        elif dimension == 9:
            pose_marching_cube(data_name, dimension, 200, model, model_key, configs, reg)
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
    if model_type == 'moe_grid':
        if dimension == 2:
            export_plot_2D_moe_grid_internal(data_name, model_key, y, reg, configs, dimension, threshold, model_detail_str)
        elif dimension in [3,4,9]:
            pass
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
    elif model_type == 'mlp':
        if dimension == 2:
            export_plot_2D_mlp_internal(data_name, model_key, y, reg, dimension, threshold, model_detail_str)
        elif dimension == 3:
            prep_openVDB(data_name, dimension, 100, model, model_key, configs, reg)
            marching_cube(data_name, dimension, 100, model, model_key, configs, reg)
        elif dimension == 9:
            pose_marching_cube(data_name, dimension, 200, model, model_key, configs, reg)
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Google sheet sync results
    gsheet_log_results(model_key, dimension,  reg, configs, data_name)