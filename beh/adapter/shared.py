import jax
import jax.numpy as jnp

def checkpoint_training_data(x : jax.Array, y : jax.Array):
    '''
    Prints a collection of relevant metrics about the training data such as datatype, range, number of samples etc.
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
    print("==================================================")

    # Catch wrong data formatting
    if jnp.min(x) != -1.0:
        raise ValueError(f"Training data X has not lower bound at -1.0")
    elif jnp.max(x) != 1.0:
        raise ValueError(f"Training data X has not upper bound at 1.0")
    
    elif jnp.min(y) != 0.0:
        raise ValueError(f"Training data labels Y have not lower bound at 0.0")
    elif jnp.max(y) != 1.0:
        raise ValueError(f"Training data labels Y have not upper bound at 1.0")
    
    else:
        pass