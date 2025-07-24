import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from datetime import datetime

from beh.registry import *
from beh.styler.registry import *

def export_plot_training_data (x : jax.Array, y : jax.Array):
    
    # Retrieve image width and height from data
    xdim = jnp.unique(x[...,0]).size
    ydim = jnp.unique(x[...,1]).size

    # Create image using label data and export
    img = y.reshape((xdim,ydim))
    dimensions = 2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = result_dir_registry[dimensions] + f"/dim2_y_{timestamp}.png"
    plt.imsave(
        path,
        img,
        cmap = cmap
    )
    print(f"2D training data plot saved at {path}")
    pass