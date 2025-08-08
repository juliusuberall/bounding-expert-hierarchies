import jax.numpy as jnp
import numpy as np
import os
import trimesh as tm

from beh.core.registry import *
from beh.registry import *
from beh.adapter.shared import blenderXYZ_to_trimeshXYZ, boundingbox_normalization

def preprocess_4D(args):
    '''
    4D 
    \nLoads the positive and negative sample created with Blender. Creates labels (y) and 4D coordinates (X, Y, Z, time). 
    \nSaves the mesh dimension and bounds to the core registry.
    '''

    # Expects a certain folder structure to pre process the 4D samples
    folder = data_dir_registry[4]
    folderN = folder + f"/{args.data_name}/negative"
    folderP = folder + f"/{args.data_name}/positive"

    ## Check if positive and negative frame amount matches 
    p_endF, n_endF = np.sum([f.split(".")[1] == "npy" for f in os.listdir(folderP)]), np.sum([f.split(".")[1] == "npy" for f in os.listdir(folderN)])
    assert p_endF == n_endF, f"positive and negatives sample frame count need to match. P:{p_endF} N:{n_endF}"

    # Loop over all frames and create labeled sample with 5D (time, X, Y, Z, label)
    t = np.linspace(0, 1, p_endF)
    sample_types = [folderN, folderP] # Index equals sample label
    x, y = [], []
    for i in range(p_endF):
        for j in range(len(sample_types)):
            samples = np.load(sample_types[j] + f"/frame_{i+1:04d}.npy")
            samples = np.concatenate((
                np.full((samples.shape[0],1), t[i]), 
                samples, 
                np.full((samples.shape[0],1), j)), 
                axis=1
            )
            x.append(samples)

        # Print when positive and negative samples of frame are processed
        end = '\n' if i == (p_endF - 1) else '\r'
        print(f'Loaded: frame_{i+1:04d}.npy', end = end, flush = True)
    
    # Pre-Process data such that
    ## x shape [n samples, 4D (frame, XYZ)]
    ## y shape [n samples, 1D (in / out)]
    x = np.stack(x,axis=0).reshape((-1,5))
    y = x[...,-1].reshape(-1).astype(jnp.float32)
    x = x[...,0:4]
    x[...,1:4] = blenderXYZ_to_trimeshXYZ(x[...,1:4]).astype(jnp.float32)

    # Get total bounding box dimensions
    # -> to translaye the samples correct after normalization
    # -> hint at sample density 
    cloud = tm.PointCloud(x[...,1:4])
    size = np.array((
        cloud.bounds[1,0] - cloud.bounds[0,0],
        cloud.bounds[1,1] - cloud.bounds[0,1],
        cloud.bounds[1,2] - cloud.bounds[0,2]))
    bounds = cloud.bounds

    return x, y, size, bounds

def load_samples(path : str, reg : CoreRegistry,):
    '''
    Load samples from a .npz formatted file with the expected content.
    '''
    # Load moe .npz file 
    loaded = np.load(path)
    
    # Unpack file
    x = jnp.array(loaded['x'])
    y = jnp.array(loaded['y'])
    size = jnp.array(loaded['size'])
    bounds = jnp.array(loaded['bounds'])

    # Normalize coordinates to range -1.0 to 1.0
    x = (x.at[...,1:4].subtract(bounds[0])).at[...,1:4].divide(size)
    x = x * 2 - 1 # Also remap time from 0.0 - 1.0 to -1.0 - 1.0

    # Store sample size per dimension 
    reg.add(core_keys['data_size_key'], size)

    return reg, x ,y
