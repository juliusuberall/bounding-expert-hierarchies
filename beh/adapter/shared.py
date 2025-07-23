import jax
import jax.numpy as jnp

def remap(value, low1, high1, low2, high2):
    # Remap values from one domain to another
    ## Taken from: https://stackoverflow.com/questions/3451553/value-remapping
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def checkpoint_training_data(x:jax.Array, y:jax.Array):
    # Checkpoint to catch numeric and type missmatches which are
    # difficult to debug on lower levels
    print(f"JAX Accelerator: {jax.devices()[0].device_kind}")
    print(f"Total Samples: {y.shape}")
    print(f'X Range: {jnp.min(x)}-{jnp.max(x)} | X Dtype: {x.dtype} | X Type: {type(x)}') 
    print(f'Y Range: {jnp.min(y)}-{jnp.max(y)} | Y Dtype: {y.dtype} | Y Type: {type(y)}')
    print(f'X[30] Sample: {x[30]}') 
    print(f'Y[30] Sample: {y[30]}')
    pass