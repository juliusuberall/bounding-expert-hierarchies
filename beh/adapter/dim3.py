import jax.numpy as jnp
import numpy as np
import trimesh as tm

from beh.registry import *
from beh.core.registry import *

def sample_pts_random (args):
    '''
    3D 
    \nLoads OBj mesh and point samples the mesh within its bounding box randomly uniform. Creates labels (y) and 3D coordinates (x). 
    '''
    # Load OBJ mesh
    path = data_dir_registry[args.dim] + f"/{args.data_name}.obj"
    mesh = tm.load_mesh(path)

    # Get bounding box dimensions
    # -> to generate uniform samples in the bounding box
    size = np.array((
        mesh.bounds[1,0] - mesh.bounds[0,0],
        mesh.bounds[1,1] - mesh.bounds[0,1],
        mesh.bounds[1,2] - mesh.bounds[0,2]))
    bounds = mesh.bounds

    # Sample
    n_samples = args.res**3
    x = np.random.uniform(size=(n_samples ,3))
    x_absolut = x * size + bounds[0]
    y = mesh.contains(x_absolut).astype(np.float32)

    return x, y, size, bounds

#------------------------------------------------------------------------------------

def sample_rays_random (args):
    '''
    3D 
    \nLoads OBj mesh and ray-samples the mesh within its bounding box randomly uniform. 
    \nCreates labels (y) and 6D encoding of ray (x) using 3D origin and 3D direction vector as in NeuralBounding (Liu et al. 2024). 
    '''
    # Load OBJ mesh
    path = data_dir_registry[args.dim] + f"/{args.data_name}.obj"
    mesh = tm.load_mesh(path)

    # Get random points for ray origin
    x, y, size, bounds = sample_pts_random(args)
    ## Returns normalized coordinates and we need absolut
    x_abs = x * size + bounds[0]

    # Get random directions for rays 
    n = x.shape[0]
    d = np.random.normal(size=(n,3))
    d = d / np.linalg.norm(d, axis=1)[:,None]

    # Sample object with ray
    y = mesh.ray.intersects_any(x_abs, d).astype(np.float32)

    # Create ray encodings
    x = np.concatenate((x, d), axis=1)

    return x, y, size, bounds

#------------------------------------------------------------------------------------

def sample_pts_grid (args):
    '''
    3D 
    \nLoads OBj mesh and samples the mesh within its bounding box in a grid. Creates labels (y) and 3D coordinates (x). 
    '''
    # Load OBJ mesh
    path = data_dir_registry[args.dim] + f"/{args.data_name}.obj"
    mesh = tm.load_mesh(path)

    # Get bounding box dimensions
    # -> to generate uniform samples in the bounding box
    size = np.array((
        mesh.bounds[1,0] - mesh.bounds[0,0],
        mesh.bounds[1,1] - mesh.bounds[0,1],
        mesh.bounds[1,2] - mesh.bounds[0,2]))
    bounds = mesh.bounds

    # Sample
    res = args.res
    lin = np.linspace(0, 1, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1)
    x = points.reshape(-1, 3) 
    x_absolut = x * size + bounds[0] # Translate with lowest corner to correct place in space
    y = mesh.contains(x_absolut).astype(np.float32)

    return x, y, size, bounds

#------------------------------------------------------------------------------------

def load_samples(path : str, reg : CoreRegistry, query : str):
    '''
    3D
    Load samples from a .npz formatted file with the expected content and update core registry.
    '''
    # Load .npz file 
    loaded = np.load(path)
    
    # Unpack file
    x = jnp.array(loaded['x'])
    y = jnp.array(loaded['y'])
    size = jnp.array(loaded['size'])
    bounds = jnp.array(loaded['bounds'])

    # Normalize coordinates to range -1.0 to 1.0 
    if query == "ray":
        d = x[...,3:6]
        x = x.at[...,:3].multiply(2).at[...,:3].subtract(1) 
    else :
        x = x * 2 - 1 

    # Store sample size per dimension 
    reg.add(core_keys['data_size_key'], size)
    reg.add(core_keys['data_bounds_key'], bounds)

    return reg, x ,y