import jax

from beh.styler.shared import *
from beh.styler.dim2 import *
from beh.core.registry import *

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
    model_key : str,
    x : jax.Array, 
    y : jax.Array, 
    reg : CoreRegistry, 
    configs : dict,
    dimension : int):
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
    export_plot_training_metrics(model_key, model_detail_str, reg, configs, dimension)

    # Create model internal state overview
    if model_type == 'moe':
        if dimension == 2:
            export_plot_2D_moe_internal_8_experts(model_key, x, y, reg, configs, dimension, threshold, model_detail_str)
        elif dimension == 3:
            pass
        elif dimension == 4:
            pass
        elif dimension == 9:
            pass
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
    elif model_type == 'mlp':
        if dimension == 2:
            pass
        elif dimension == 3:
            pass
        elif dimension == 4:
            pass
        elif dimension == 9:
            pass
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")