import jax.numpy as jnp
import numpy as np

from beh.core.registry import *
from beh.registry import *
from PIL import Image
import trimesh as tm
from skimage.measure import marching_cubes

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
    # If not already binary, convert alpha channel to binary
    y = ((img / 255.0) > 0.1).astype(int) 
    y = jnp.array(y, dtype=jnp.float32)[...,3].flatten()
    x = jnp.indices(img[...,3].shape).reshape(2,-1).T.astype(jnp.float32)

    # Normalize coordinates to range -1.0 to 1.0
    x = x.at[...,0].divide(img.shape[0] - 1)
    x = x.at[...,1].divide(img.shape[1] - 1)
    x = x * 2 - 1 

    # Store image resolution
    reg.add(core_keys['data_size_key'], img.shape)

    return reg, x, y

#------------------------------------------------------------------------------------

def sample_rays_random (args):
    '''
    2D 
    \nLoads image and ray-samples the iso-surface reconstructred mesh of the alpha mask in 3D within its bounding box randomly uniform. 
    \nCreates labels (y) and 4D encoding of ray (x) using 2D origin and 2D direction vector. 
    '''
    # Load image and alpha mask
    img = data_dir_registry[args.dim] + f"/{args.data_name}.png"
    img = Image.open(img)
    alpha = np.asarray(img, dtype=np.float32)[...,-1]
    alpha = ((alpha / 255.0) > 0.1).astype(float)

    # Use marching cubes to create a 3D voxel representation of the alpha mask that we can ray cast
    padding = np.zeros_like(alpha)
    alpha = np.stack([padding,alpha,padding], axis=-1)
    verts, faces, normals, values_on_surface = marching_cubes(alpha, level=0.5, spacing=(1, 1, 1))
    mesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=-normals)

    # Get random points for ray origin
    ## Assuming square image with "width = height"
    width = height = alpha.shape[0]
    size = np.array((width, height, 0))
    n_samples = args.res ** 2
    x = np.random.uniform(0, 1, (n_samples, 2))
    ### Set all points in plane for raycasting with mesh 
    x_abs = x * width
    x_abs = np.concatenate([x_abs, np.ones((x_abs.shape[0], 1))], axis=-1)

    # Get random directions for rays 
    n = x.shape[0]
    d = np.random.normal(size=(n,2))
    d = d / np.linalg.norm(d, axis=1)[:,None]
    d3 = np.concatenate([d, np.ones((d.shape[0], 1))], axis=-1)

    # Sample object with ray
    y = mesh.ray.intersects_any(x_abs, d3).astype(np.float32)

    # Create ray encodings
    x = np.concatenate((x, d), axis=1)

    return x, y, size, 0

#------------------------------------------------------------------------------------

def load_samples(path : str, reg : CoreRegistry, query : str, dim : int, data_name : str):
    '''
    2D
    Load samples from a .npz formatted file with the expected content and update core registry.
    '''
    if query == "point":
        return preprocess_rgba(data_dir_registry[dim] + f"/{data_name}.png", reg, False)
    else:
        npz = np.load(path)
        x = jnp.array(npz['x'])
        y = jnp.array(npz['y'])
        size = jnp.array(npz['size'])
        bounds = jnp.array(npz['bounds'])

        # Normalize point coordinates to range -1.0 to 1.0 
        x = x.at[...,:2].multiply(2).at[...,:2].subtract(1) 

        # Store sample size per dimension 
        reg.add(core_keys['data_size_key'], size)
        reg.add(core_keys['data_bounds_key'], bounds)

        return reg, x ,y

