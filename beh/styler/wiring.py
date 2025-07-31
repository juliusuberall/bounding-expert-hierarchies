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
    elif dimension == 10:
        pass
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")

def format_export_results(x : jax.Array, y : jax.Array, reg : CoreRegistry, configs : dict, dimension : int):
    '''
    Create result plots.
    \nDelegates to corresponding data dimensionality sub-routine.
    '''
    # Get general config
    threshold = configs['general']['boundary_threshold']
    
    # Result plots for all dimensions
    export_plot_training_metrics(reg=reg,dimension=dimension)

    if dimension == 2:
        export_plot_2D_moe_internal_8_experts(x, y, reg, configs, dimension, threshold)
        pass
    elif dimension == 3:
        pass
    elif dimension == 4:
        pass
    elif dimension == 10:
        pass
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")