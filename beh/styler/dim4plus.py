import numpy as np
from skimage.measure import marching_cubes

from beh.core.registry import *
from beh.core.shared import batch_data
from beh.core.moe import batch_query_moe_OOM
from beh.core.moe_sparse import *

inf_batch_size = 2048 # Important to ensure no query swalloing when sampling full MoE

#------------------------------------------------------------------------------------

def marching_cube_9D(
    pose_num : int,
    pose ,
    data_name : str,
    dimension : int, 
    grid_res : int,
    model ,
    model_key : str,
    configs : dict,
    reg : CoreRegistry):
    '''Create a .ply file for each selected pose with the extracted decision boundary using marching cubes in a MoE.'''

    # Get data info
    threshold = configs['general']['boundary_threshold']
    dim = reg.get(core_keys['data_size_key'])
    bounds = reg.get(core_keys['data_bounds_key'])

    # Define sampling grid for marching cubes
    ## Defines mesh resolution
    mc_res = grid_res
    lin = np.linspace(-1, 1, mc_res)
    mc_x, mc_y, mc_z = np.meshgrid(lin, lin, lin, indexing='ij')  # use 'ij' for matrix-style indexing
    pose_cube = np.array([np.full((grid_res, grid_res, grid_res), v) for v in pose])
    mc_x = np.stack([mc_x, mc_y, mc_z])
    mc_x = np.concatenate((mc_x,pose_cube)).reshape((9,-1)).T

    # Evaluate the function over grid
    mc_x = batch_data(mc_x, inf_batch_size)
    values = batch_query_moe_OOM(mc_x, model, sparse_funcs_2048[model_key], 50)
    values = np.array(values).reshape((mc_res,mc_res,mc_res))

    # Extract mesh at with threshold of SDF
    voxel_spacing = (dim[0]/mc_res, dim[1]/mc_res, dim[2]/mc_res)
    verts, faces, normals, values_on_surface = marching_cubes(values, level=threshold, spacing=voxel_spacing)

    # Export
    implicitMesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    implicitMesh.apply_translation(bounds[0])
    path = result_visual_registry[dimension] + '/'+ data_name + "_P" + str(pose_num) + "_" + model_key
    implicitMesh.export(f"{path}.ply")

#------------------------------------------------------------------------------------

def pose_marching_cube (
    data_name : str,
    dimension : int, 
    grid_res : int,
    model ,
    model_key : str,
    configs : dict,
    reg : CoreRegistry):
    '''Loop over specified 6D poses and extract the decision boundary in a MoE.'''
    
    # Select poses to query
    ## This hardcoded and specific to the dataset. If creating new
    ## dataset for a robot this needs to be updated.
    poses = np.array([
        [2.27260541021,-1.61643156397,2.16519718803,-2.11956195086,-1.57079632679,0.701809083412], # frame 4
        [2.84013220260,-1.31661436222,1.82411728984,-2.07829925441,-1.57079632679,1.26933587580], # frame 8
        [3.22525375718,-1.44616173088,2.24267958040,-2.36731417631,-1.57079632679,1.65445743038], # frame 12
        [2.65759882087,-1.42482560108,2.25287529317,-2.39884601889,-1.57079632679,1.08680249408] # frame 16
    ])

    # Extract mesh for poses
    for i in range(poses.shape[0]):
        marching_cube_9D(
            pose_num=i,
            pose=poses[i],
            data_name=data_name,
            dimension=dimension,
            grid_res=grid_res,
            model=model,
            model_key=model_key,
            configs=configs,
            reg=reg
        )