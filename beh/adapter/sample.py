import numpy as np
import jax.numpy as jnp

from beh.core.registry import *
from beh.registry import *

def checkpoint_samples(x, y, size):
    '''
    Prints a collection of relevant metrics about the sampled samples such as datatype, range, number of samples etc.
    '''
    # Checkpoint to catch numeric and type missmatches which are
    # difficult to debug on lower levels
    print("================================================== Pre-Process Checkpoint")
    print(f"Total Samples: {y.shape}")
    print(f"Size: X {size[0]:04f} Y {size[1]:04f} Z {size[2]:04f}")
    print(f'X Range: {np.min(x):04f} - {np.max(x):04f} | X Dtype: {x.dtype} | X Type: {type(x)} | X Dims: {x.shape[1]}') 
    print(f'Y Range: {np.min(y):04f} - {np.max(y):04f} | Y Dtype: {y.dtype} | Y Type: {type(y)}')
    print(f'X[30] Sample: {x[30]}') 
    print(f'Y[30] Sample: {y[30]}')

def save_samples(args, x, y, size, bounds):
    '''
    Saves the samples as .npz file with keywords x, y, size, bounds and strategyt(sampling).
    '''
    if x.shape[0] < 1e+3 :
        n_samples = x.shape[0]
    elif x.shape[0] < 1e+6 :
        n_samples = f"{int(x.shape[0] / 1000)}K"
    else:
        n_samples = f"{int(x.shape[0] / 1e+6)}M"
    
    if args.dim == 3 :
        path = data_dir_registry[args.dim] + f"/{args.data_name}_{args.query}_{n_samples}_{args.strategy}_samples.npz"
    else :
        path = data_dir_registry[args.dim] + f"/{args.data_name}_{args.query}_{args.start}to{args.end}_{args.frame_rate}f_{n_samples}_samples.npz"

    np.savez(
        path,
        x = x,
        y = y,
        size = size,
        bounds = bounds,
        strategy = args.strategy,
    )
    return path