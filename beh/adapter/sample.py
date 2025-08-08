import numpy as np
import jax.numpy as jnp

from beh.core.registry import *
from beh.registry import *

def checkpoint_samples(x, y, size):
    '''
    Prints a collection of relevant metrics about the training data such as datatype, range, number of samples etc.
    '''
    # Checkpoint to catch numeric and type missmatches which are
    # difficult to debug on lower levels
    print("================================================== Sampling Checkpoint")
    print(f"Total Samples: {y.shape}")
    print(f"Size: X {size[0]:04f} Y {size[1]:04f} Z {size[2]:04f}")
    print(f'X Range: {np.min(x):04f} - {np.max(x):04f} | X Dtype: {x.dtype} | X Type: {type(x)} | X Dims: {x.shape[1]}') 
    print(f'Y Range: {np.min(y):04f} - {np.max(y):04f} | Y Dtype: {y.dtype} | Y Type: {type(y)}')
    print(f'X[30] Sample: {x[30]}') 
    print(f'Y[30] Sample: {y[30]}')
    print("==================================================")

def save_samples(args, x, y, size, bounds):
    '''
    Saves the samples as .npz file with keywords x, y and size.
    '''
    K_samples = int(x.shape[0] / 1000)
    path = data_dir_registry[args.dim] + f"/{args.data_name}_{K_samples}K_{args.strategy}_samples.npz"
    np.savez(
        path,
        x = x,
        y = y,
        size = size,
        bounds = bounds
    )
    return path

def load_samples(path : str, reg : CoreRegistry,):
    # Load moe .npz file 
    loaded = np.load(path)
    
    # Unpack file
    x = jnp.array(loaded['x'])
    y = jnp.array(loaded['y'])
    size = jnp.array(loaded['size'])
    bounds = jnp.array(loaded['bounds'])

    # Normalize coordinates to range -1.0 to 1.0
    x = (x - bounds[0]) / size
    x = x * 2 - 1 

    # Store sample size per dimension 
    reg.add(core_keys['data_size_key'], size)

    return reg, x ,y