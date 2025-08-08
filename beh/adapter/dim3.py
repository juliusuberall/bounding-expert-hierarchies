import jax.numpy as jnp
import numpy as np
import trimesh as tm

from beh.registry import *
from beh.core.registry import *

def sample_obj_random (args):
    '''
    3D 
    \nLoads OBj mesh and samples the mesh within its bounding box randomly uniform. Creates labels (y) and 3D coordinates (x). 
    \nSaves the mesh dimension to the core registry.
    '''
    # Load OBJ mesh
    path = data_dir_registry[args.dim] + f"/{args.data_name}.obj"
    mesh = tm.load_mesh(path)

    # Get bounding box dimensions
    # -> to generate uniform samples in the bounding box
    # -> hint at sampling density 
    size = np.array((
        mesh.bounds[1,0] - mesh.bounds[0,0],
        mesh.bounds[1,1] - mesh.bounds[0,1],
        mesh.bounds[1,2] - mesh.bounds[0,2]))
    bounds = mesh.bounds

    # Sample
    n_samples = args.res**3
    x = np.random.uniform(size=(n_samples ,3)) * size + bounds[0]
    y = mesh.contains(x).astype(np.float32)

    return x, y, size, bounds

def sample_obj_grid (args):
    '''
    3D 
    \nLoads OBj mesh and samples the mesh within its bounding box in a grid. Creates labels (y) and 3D coordinates (x). 
    \nSaves the mesh dimension and bounds to the core registry.
    '''
    # Load OBJ mesh
    path = data_dir_registry[args.dim] + f"/{args.data_name}.obj"
    mesh = tm.load_mesh(path)

    # Get bounding box dimensions
    # -> to generate uniform samples in the bounding box
    # -> hint at sampling density 
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
    x = (x - bounds[0]) / size
    x = x * 2 - 1 

    # Store sample size per dimension 
    reg.add(core_keys['data_size_key'], size)

    return reg, x ,y
