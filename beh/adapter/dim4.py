import jax.numpy as jnp
import numpy as np
import os
import trimesh as tm

from beh.core.registry import *
from beh.registry import *
from beh.adapter.shared import blenderXYZ_to_trimeshXYZ

def preprocess_4D_points(args):
    '''
    4D 
    \nLoads the positive and negative sample created with Blender. Creates labels (y) and 4D coordinates (X, Y, Z, time). 
    '''

    # Expects a certain folder structure to pre process the 4D samples
    folder = data_dir_registry[4]
    folderN = folder + f"/{args.data_name}/negative"
    folderP = folder + f"/{args.data_name}/positive"
    start_frame, end_frame = args.start, args.end
    frame_rate = args.frame_rate

    ## Check if positive and negative frame amount matches 
    p_endF, n_endF = np.sum([f.split(".")[1] == "npy" for f in os.listdir(folderP)]), np.sum([f.split(".")[1] == "npy" for f in os.listdir(folderN)])
    assert p_endF == n_endF, f"positive and negatives sample frame count need to match. P:{p_endF} N:{n_endF}"

    # Loop over all selected frames and create labeled sample with 5D (time, X, Y, Z, label)
    t = np.linspace(0, 1, end_frame - start_frame)
    sample_types = [folderN, folderP] # Index equals sample label
    x, y = [], []
    for i in range(start_frame, end_frame, frame_rate):
        for j in range(len(sample_types)):
            samples = np.load(sample_types[j] + f"/frame_{i:04d}.npy")
            samples = np.concatenate((
                np.full((samples.shape[0],1), t[i - start_frame]), 
                samples, 
                np.full((samples.shape[0],1), j)), 
                axis=1
            )
            x.append(samples)

        # Print when positive and negative samples of frame are processed
        end = '\n' if i == (end_frame - 1) else '\r'
        print(f'Loaded: frame_{i:04d}.npy', end = end, flush = True)
    
    # Pre-Process data such that
    ## x shape [n samples, 4D (frame, XYZ)]
    ## y shape [n samples, 1D (in / out)]
    x = np.stack(x,axis=0).reshape((-1,5)).astype(jnp.float32)
    y = x[...,-1].reshape(-1).astype(jnp.float32)
    x = x[...,0:4]
    x[...,1:4] = blenderXYZ_to_trimeshXYZ(x[...,1:4]).astype(jnp.float32)

    # Get total bounding box dimensions
    # -> to translate the samples correct after normalization for the model
    cloud = tm.PointCloud(x[...,1:4])
    size = np.array((
        cloud.bounds[1,0] - cloud.bounds[0,0],
        cloud.bounds[1,1] - cloud.bounds[0,1],
        cloud.bounds[1,2] - cloud.bounds[0,2]))
    bounds = cloud.bounds

    return x, y, size, bounds

#------------------------------------------------------------------------------------

def sample_rays_random(args):
    '''
    4D
    \nLoads a and ray-samples an animation in 3D expecting for each frame a .obj mesh.
    \nCreates samples x (7D input vector) and y (1D label) 
    \n\nargs.res = number of rays sampled per frame
    '''
    # Get general parameters
    folder = f'{data_dir_registry[args.dim]}/{args.data_name}/'
    prefix = args.data_name
    n = args.res
    start_frame, end_frame = args.start, args.end

    # Get bounding box of last frame
    ## Since it is a fluid we assume last frame will have max. dimensions encapsulating everything prior
    frame_obj_path = folder + prefix + f'_{end_frame-1:04d}.obj'
    max_mesh = tm.load_mesh(frame_obj_path)
    bb = max_mesh.bounding_box

    # Loop over all selected frames and create labeled sample for 7D (time, XYZ-origin, XYZ-direction)
    t = np.linspace(0, 1, end_frame - start_frame)
    X, Y = [], []

    for frame in range(start_frame, end_frame):
        # Load mesh
        frame_obj_path = folder + prefix + f'_{frame:04d}.obj'
        mesh = tm.load_mesh(frame_obj_path)

        # Sample n random ray origins
        x = np.random.uniform(0, 1, size=(n, 3))
        ## Compute absolut positions in bounding box and move from min coordinate into space 
        min_coord, size = bb.bounds[0], bb.extents
        x_abs = x * size + min_coord
        ## Compute random direction vector
        d = np.random.normal(size=(n,3))
        d = d / np.linalg.norm(d, axis=1)[:,None]

        # Sample frame mesh with ray
        y = mesh.ray.intersects_any(x_abs, d).astype(np.float32)
        y = np.expand_dims( y, axis=-1)

        # Create ray encodings
        x = np.concatenate((x, d), axis=1)

        # combine to 7D (time, XYZ-origin, XYZ-direction)
        x = np.concatenate((
            np.full((x.shape[0],1), t[frame - start_frame]), 
            x), 
            axis=1
        )
        X.append(x)
        Y.append(y)

        # Print when positive and negative samples of frame are processed
        end = '\n' if frame == (end_frame - 1) else '\r'
        print(f'Loaded: frame_{frame:04d}.npy', end = end, flush = True)

    X = np.concatenate(X,axis=0)
    Y = np.concatenate(Y,axis=0)
    x = np.array(X)
    y = np.array(Y)

    # Get total bounding box dimensions
    # -> to translate the samples correct after normalization for the model
    size = np.array((
        bb.bounds[1,0] - bb.bounds[0,0],
        bb.bounds[1,1] - bb.bounds[0,1],
        bb.bounds[1,2] - bb.bounds[0,2]))
    bounds = bb.bounds

    return x, y, size, bounds

#------------------------------------------------------------------------------------

def load_samples(path : str, reg : CoreRegistry, query : str):
    '''
    4D
    Load samples from a .npz formatted file with the expected content and update core registry.
    '''
    # Load moe .npz file 
    loaded = np.load(path)
    
    # Unpack file
    x = jnp.array(loaded['x'])
    y = jnp.array(loaded['y'].flatten()) # WRONG SHAPE IN DATASET !!! y needs to be flat originally
    size = jnp.array(loaded['size'])
    bounds = jnp.array(loaded['bounds'])

    # Normalize coordinates to range -1.0 to 1.0
    if query == "point":
        x = (x.at[...,1:4].subtract(bounds[0])).at[...,1:4].divide(size)
        x = x * 2 - 1 # Also remap time from 0.0 - 1.0 to -1.0 - 1.0
    elif query == "ray" :
        x = x.at[...,:4].multiply(2).at[...,:4].subtract(1)  
    else:
        raise ValueError(f"Unsupported query type: {query}")


    # Store sample size per dimension 
    reg.add(core_keys['data_size_key'], size)
    reg.add(core_keys['data_bounds_key'], bounds)

    return reg, x ,y
