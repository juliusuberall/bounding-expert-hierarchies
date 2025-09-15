import numpy as np
import jax.numpy as jnp

from beh.registry import *
from beh.core.registry import *
from beh.core.moe_sparse import *
from beh.core.mlp import mlp_forward_INF
from beh.core.shared import batch_data
from beh.core.moe_benchmarking import gating_confidence

#------------------------------------------------------------------------------------

def prep_openVDB(
    dimension : int, 
    vdb_res : int,
    model ,
    model_key : str,
    configs : dict,
    reg : CoreRegistry):
    '''Create a .npy file that we can use in Blender to simply convert into a openVDB as Blender ships with the functions for this.'''

    print("\nExtracting results and formatting for OpenVDB.")

    # Get configs
    model_type = configs[model_key]['type']
    dim = reg.get(core_keys['data_size_key'])
    bounds = reg.get(core_keys['data_bounds_key'])

    # Define sampling grid for OpenVDB volume
    lin = jnp.linspace(-1, 1, vdb_res)
    vdb_X, vdb_Y, vdb_Z = jnp.meshgrid(lin, lin, lin, indexing='ij')  # use 'ij' for matrix-style indexing
    vdb_x = jnp.stack([vdb_X, vdb_Y, vdb_Z], axis=-1).reshape(-1, 3) 

    # Break in batches
    inf_batch_size = 200000 # Hardcoded in project 
    x_batches = batch_data(vdb_x, inf_batch_size)
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])

    if model_type == 'moe':
        nex = configs[model_key]['nex']
        # Forward through model and compute voxel densities
        func = sparse_funcs_200K[model_key]
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

    path = result_vdb_dir_registry[dimension] + '/'+ model_key
    np.savez(f"{path}.npz",
                activation = activation, 
                density = density,
                origin_offset = origin_offset,
                voxel_spacing = voxel_spacing,
                mat_dix = mat_dix)
    print(f"Density range: {float(np.min(density)):.2f} - {float(np.max(density)):.2f}")
    print(f"Exported .npz at {path}")