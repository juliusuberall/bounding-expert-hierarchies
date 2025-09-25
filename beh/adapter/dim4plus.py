import jax.numpy as jnp
import numpy as np
import os
import trimesh as tm

from beh.core.registry import *
from beh.registry import *
from beh.adapter.shared import blenderXYZ_to_trimeshXYZ
from beh.core.shared import remap

def preprocess_4Dplus(args):
    '''
    4D plus 
    \nLoads the positive and negative samples for a universal robot 10 created with Grasshopper. 
    \nCreates labels (y) and 9D coordinates (X ,Y ,Z ,RJ1 ,RJ2 ,RJ3 ,RJ4 ,RJ5 ,RJ6 ). 
    \nSaves the mesh dimension and bounds to the core registry.
    '''

    # Expects a certain folder structure to pre process the 4D samples
    folder = data_dir_registry[9]
    folderN = folder + f"/{args.data_name}/negative"
    folderP = folder + f"/{args.data_name}/positive"

    ## Check if positive and negative frame amount matches 
    p_endF, n_endF = np.sum([f.split(".")[1] == "txt" for f in os.listdir(folderP)]), np.sum([f.split(".")[1] == "txt" for f in os.listdir(folderN)])
    assert p_endF == n_endF, f"positive and negatives sample frame count need to match. P:{p_endF} N:{n_endF}"

    # Loop over all frames and create labeled sample with 10D (XYZ, RJ1-6, label)
    sample_types = [folderN, folderP] # Index equals sample label
    x, y = [], []
    for i in range(p_endF):
        frame_samples = []
        for j in range(len(sample_types)):
            samples = np.loadtxt(sample_types[j] + f"/frame_{i+1:04d}.txt", delimiter=",")
            samples = np.concatenate((
                samples, 
                np.full((samples.shape[0],1), j)), 
                axis=1
            )
            frame_samples.append(samples)
        x.append(np.concatenate((frame_samples),axis=0))

        # Print when positive and negative samples of frame are processed
        end = '\n' if i == (p_endF - 1) else '\r'
        print(f'Loaded: frame_{i+1:04d}.npy', end = end, flush = True)
    
    # Pre-Process data such that
    ## x shape [n samples, 9D (XYZ, RJ1-6)]
    ## y shape [n samples, 1D (in / out)]
    x = np.stack(x,axis=0).reshape((-1,10)).astype(jnp.float32)
    y = x[...,-1].reshape(-1).astype(jnp.float32)
    x = x[...,:9]
    x[...,:3] = blenderXYZ_to_trimeshXYZ(x[...,:3]) # Not sure if thats needed here

    # Get total bounding box dimensions
    # -> to translate the samples correct after normalization
    # -> hint at sample density 
    cloud = tm.PointCloud(x[...,:3])
    size = np.array((
        cloud.bounds[1,0] - cloud.bounds[0,0],
        cloud.bounds[1,1] - cloud.bounds[0,1],
        cloud.bounds[1,2] - cloud.bounds[0,2]))
    bounds = cloud.bounds

    return x, y, size, bounds


def load_samples(path : str, reg : CoreRegistry,):
    '''
    4D plus
    Load samples from a .npz formatted file with the expected content.
    '''
    # Load moe .npz file 
    loaded = np.load(path)
    
    # Unpack file
    x = loaded['x']
    y = jnp.array(loaded['y'])
    size = jnp.array(loaded['size'])
    bounds = jnp.array(loaded['bounds'])

    # Normalize to range -1.0 to 1.0
    ## Points
    x[...,:3] = (x[...,:3] - bounds[0]) / size
    ## Joints - expected to have range -2*Pi to 2*Pi
    x[...,3:] = remap(
        x[...,3:],
        -2*np.pi,
        2*np.pi,
        0,
        1
    )
    x = jnp.array(x * 2 - 1)

    # Store sample size per dimension 
    reg.add(core_keys['data_size_key'], size)
    reg.add(core_keys['data_bounds_key'], bounds)

    return reg, x ,y