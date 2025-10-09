import jax
import jax.numpy as jnp
import numpy as np
import trimesh as tm

from beh.core.shared import remap

def checkpoint_training_data(x : jax.Array, y : jax.Array):
    '''
    Prints a collection of relevant metrics about the training data such as datatype, range, number of samples etc. and throws error if out of range.
    '''
    # Checkpoint to catch numeric and type missmatches which are
    # difficult to debug on lower levels
    print("================================================== Training Data Checkpoint")
    print(f"JAX Accelerator: {jax.devices()[0].device_kind}")
    print(f"Total Samples: {y.shape}")
    print(f'X Range: {jnp.min(x)} - {jnp.max(x)} | X Dtype: {x.dtype} | X Type: {type(x)} | X Dims: {x.shape[1]}') 
    print(f'Y Range: {jnp.min(y)} - {jnp.max(y)} | Y Dtype: {y.dtype} | Y Type: {type(y)}')
    print(f'X[30] Sample: {x[30]}') 
    print(f'Y[30] Sample: {y[30]}')

    # Catch wrong data formatting
    if jnp.min(x) < -1.0:
        raise ValueError(f"Training data X lower bound exceeds -1.0")
    elif jnp.max(x) > 1.0:
        raise ValueError(f"Training data X upper bound exceeds 1.0")
    
    elif jnp.min(y) != 0.0:
        raise ValueError(f"Training data labels Y have not lower bound at 0.0")
    elif jnp.max(y) != 1.0:
        raise ValueError(f"Training data labels Y have not upper bound at 1.0")
    
    else:
        pass

# Helper functions
def blenderXYZ_to_trimeshXYZ(p):
    # Convert Blender coordinate system to Trimesh system
    ## Swap Y and Z axes to convert to Trimesh-style orientation
    ## Flip the new Z axis (original Y in Blender)
    p = p[:, [0, 2, 1]]
    p[:, 2] *= -1 
    return p

def boundingbox_normalization(x):
    # Normalize set of points based on their enclosing bounding box
    cloud = tm.PointCloud(x)
    x =  remap(x,
          cloud.bounds[0],
          cloud.bounds[1],
          np.array([0,0,0]),
          np.array([1,1,1]),
          )
    bb0 = cloud.bounds[1,0] - cloud.bounds[0,0]
    bb1 = cloud.bounds[1,1] - cloud.bounds[0,1]
    bb2 = cloud.bounds[1,2] - cloud.bounds[0,2]
    return x, bb0 , bb1 , bb2, cloud.bounds[0,...]