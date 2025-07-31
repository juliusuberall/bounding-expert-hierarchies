import jax
import jax.numpy as jnp

def get_fn_fp_rate(yp : jax.Array, y : jax.Array, threshold : float = 0.1):
    '''
    Measure conservativness of predictions and labels with False-Negative and False-Positive rate.
    '''
    fp_rate = jnp.sum((yp > threshold) * ( y == 0)) / y.size
    fn_rate = jnp.sum((yp < threshold) * ( y > 0)) / y.size
    return fn_rate, fp_rate