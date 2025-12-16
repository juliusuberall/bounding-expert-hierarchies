import numpy as np
import jax.numpy as jnp
import trimesh as tm
from PIL import Image

from beh.registry import *
from beh.core.shared import remap
from beh.core.registry import *
from beh.core.moe import batch_query_moe_OOM, moe_forward_sparse_INF
from beh.core.mlp import mlp_forward_INF, batch_query_mlp_OOM
from beh.core.shared import batch_data
from beh.core.moe_benchmarking import gating_confidence
from skimage.measure import marching_cubes

inf_batch_size = 2048 # Important to ensure no sparse MoE query swalloing when sampling full MoE

#------------------------------------------------------------------------------------

def marching_cube(
    data_name : str,
    dimension : int, 
    grid_res : int,
    model ,
    model_key : str,
    configs : dict,
    reg : CoreRegistry):
    '''Create a .ply file for the extracted decision boundary using marching cubes.'''

    # Get data info
    threshold = configs['general']['boundary_threshold']
    dim = reg.get(core_keys['data_size_key'])
    bounds = reg.get(core_keys['data_bounds_key'])
    model_type = configs[model_key]['type']

    # Define sampling grid for marching cubes
    ## Defines mesh resolution
    mc_res = grid_res
    lin = np.linspace(-1, 1, mc_res)
    mc_x, mc_y, mc_z = np.meshgrid(lin, lin, lin, indexing='ij')  # use 'ij' for matrix-style indexing
    mc_x = np.stack([mc_x, mc_y, mc_z], axis=-1).reshape(-1, 3)  

    # Evaluate the function over grid
    mc_x = batch_data(mc_x, inf_batch_size)
    if model_type == 'moe':
        values = batch_query_moe_OOM(mc_x, model, moe_forward_sparse_INF, 50)
    else:
        values = batch_query_mlp_OOM(mc_x, model, 50)
    values = np.array(values).reshape((mc_res,mc_res,mc_res))

    # Extract mesh at with threshold of SDF
    voxel_spacing = (dim[0]/mc_res, dim[1]/mc_res, dim[2]/mc_res)
    verts, faces, normals, values_on_surface = marching_cubes(values, level=threshold, spacing=voxel_spacing)

    # Export
    implicitMesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    implicitMesh.apply_translation(bounds[0])
    path = result_visual_registry[dimension] + '/'+ data_name + "_" + model_key
    implicitMesh.export(f"{path}.ply")

#------------------------------------------------------------------------------------

def render_view(
    data_name : str,
    dimension : int, 
    model ,
    model_key : str,
    configs : dict):
    '''Render a view using the implicit field of ray intersections.'''

    print("\nRendering View")

    # Create image raster 3D points
    width, height = 1000, 1000
    x = np.linspace(-1,1, width)
    y = np.linspace(-1,1, height)
    x, y = np.meshgrid(x, y, indexing='ij')
    o = np.zeros_like(x)
    coords = np.stack([x, y, o], axis=-1).reshape((width*height,3))

    # Extend points to rays
    d = np.repeat(np.array((0,0,1))[None,:], o.size, axis=0)
    rays = np.concatenate([coords, d], axis=-1)

    # Query model
    model_type = configs[model_key]['type']
    batch_size = configs['general']['batch_size']
    rays_batched = batch_data(rays, batch_size)
    if model_type == 'moe':
        y = batch_query_moe_OOM(rays_batched, model, moe_forward_sparse_INF, 50)
    elif model_type == 'mlp':
        y = batch_query_mlp_OOM(rays_batched, model, 50)
    else :
        raise ValueError(f"Unsupported model type: {model_type}")

    # Save image
    img = y.reshape((width, height))
    img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    path = result_visual_registry[dimension] + '/'+ data_name + "_" + model_key
    img.save(f"{path}_raytraced.png")

#------------------------------------------------------------------------------------

def prep_openVDB(
    data_name : str,
    dimension : int, 
    vdb_res : int,
    model ,
    model_key : str,
    configs : dict,
    reg : CoreRegistry):
    '''Create a .npy file that we can use in Blender to simply convert into a openVDB as Blender ships with the functions for this.'''

    print("\nExtracting results and formatting for OpenVDB.")

    # Get configs and data info
    model_type = configs[model_key]['type']
    dim = reg.get(core_keys['data_size_key'])
    bounds = reg.get(core_keys['data_bounds_key'])

    # Define sampling grid for OpenVDB volume
    lin = jnp.linspace(-1, 1, vdb_res)
    vdb_X, vdb_Y, vdb_Z = jnp.meshgrid(lin, lin, lin, indexing='ij')  # use 'ij' for matrix-style indexing
    vdb_x = jnp.stack([vdb_X, vdb_Y, vdb_Z], axis=-1).reshape(-1, 3) 

    # Break in batches
    x_batches = batch_data(vdb_x, inf_batch_size)
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])

    if model_type == 'moe':
        nex = configs[model_key]['nex']
        # Forward through model and compute voxel densities
        func = moe_forward_sparse_INF
        # Originally we used vmap here but this caused for large models always OOM on a T4, 
        # which is why we swapped to a on devcice sequential compute, forward passing the batch
        # through each expert.
        def run_in_batches(func, moe, x_batches):
            def step(carry, xb):
                yp = func(moe, xb).flatten()
                return carry, yp
            _, yp = jax.lax.scan(step, None, x_batched)
            return jnp.concatenate(yp, axis=0)
        yp = run_in_batches(func, model, x_batched)
        yp_tail = func(model, x_batches[-1]).flatten()
        yp = jnp.concatenate((yp, yp_tail), axis=0)
        # Remap back in binary range
        yp = remap(
            yp, 
            jnp.min(yp),
            jnp.max(yp),
            0,
            1,)
        # Get top1 expert id per voxel
        _ , _ , expert_idx = gating_confidence(model, x_batches)
        # Assign correct material for blender import
        expert_material_table = {
            4 : 0,
            16 : 1,
            32 : 2,
        }
        mat_dix = expert_material_table[nex]

    else:
        # Forward through mlp
        func = mlp_forward_INF
        yp = jax.vmap(lambda X: func(model, X))(x_batched).flatten()
        ## Add tail
        x_tail = func(model, x_batches[-1])
        yp = jnp.concatenate((yp, x_tail.flatten()))
        # Add placeholder for moe data
        expert_idx = np.zeros((yp.shape[0]))
        # Assign correct material for blender import
        mat_dix = 3

    # Format .npz file and export
    activation = np.array(expert_idx.reshape((vdb_res,vdb_res,vdb_res)))
    density = np.array(yp.reshape((vdb_res,vdb_res,vdb_res))).astype(np.float64)
    origin_offset = bounds[0].astype(np.float64)
    voxel_spacing = np.array((dim[0]/vdb_res, dim[1]/vdb_res, dim[2]/vdb_res))

    path = result_visual_registry[dimension] + '/'+ data_name + "_" + model_key
    np.savez(f"{path}.npz",
                activation = activation, 
                density = density,
                origin_offset = origin_offset,
                voxel_spacing = voxel_spacing,
                mat_dix = mat_dix)
    print(f"Density range: {float(np.min(density)):.2f} - {float(np.max(density)):.2f}")
    print(f"Exported .npz at {path}")