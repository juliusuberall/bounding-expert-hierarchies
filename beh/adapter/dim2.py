import jax.numpy as jnp
import numpy as np

from beh.core.registry import *
from PIL import Image

def preprocess_rgba (path : str, reg : CoreRegistry, downsample : bool = True):
    '''
    2D 
    \nLoads RGBA images, extracting the alpha channel as labels (y) and normalized pixel coordinates as data (x). 
    \nSaves the image dimension to the core registry.
    '''
    # Load image
    img = Image.open(path)
    if downsample:
        img = img.resize((300, 300), resample=Image.BILINEAR) 
    img = np.asarray(img, dtype=np.float32)
    assert img.shape[1] != 4, f"2D data is expected to have 4 channel but has {img.shape[1]}."
    y = ((img / 255.0) > 0).astype(int)
    y = jnp.array(y, dtype=jnp.float32)[...,3].flatten()
    x = jnp.indices(img[...,3].shape).reshape(2,-1).T.astype(jnp.float32)

    # Normalize coordinates to range -1.0 to 1.0
    x = x.at[...,0].divide(img.shape[0] - 1)
    x = x.at[...,1].divide(img.shape[1] - 1)
    x = x * 2 - 1 

    # Store image resolution
    reg.add(core_keys['data_size_key'], img.shape)

    return reg, x, y