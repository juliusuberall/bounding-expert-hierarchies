import jax.numpy as jnp
import matplotlib.pyplot as plt
from beh.core.registry import *

def preprocess_rgba (path:str, reg:CoreRegistry):
    '''
    2D 
    \nLoads RGBA images, extracting the alpha channel as labels (y) and normalized pixel coordinates as data (x). 
    \nSaves the image dimension to the core registry.
    '''

    # Load image
    img = plt.imread(path)
    assert img.shape[1] != 4, f"2D data is expected to have 4 channel but has {img.shape[1]}."
    y = jnp.array(img[...,3].flatten(), dtype = jnp.float32)
    x = jnp.indices(img[...,3].shape).reshape(2,-1).T.astype(jnp.float32)

    # Normalize coordinates to range -1.0 to 1.0
    x = x.at[...,0].divide(img.shape[0] - 1)
    x = x.at[...,1].divide(img.shape[1] - 1)
    x = x * 2 - 1 

    # Store image resolution
    reg.add(core_keys['data_resolution_key'], img.shape)

    return reg, x, y