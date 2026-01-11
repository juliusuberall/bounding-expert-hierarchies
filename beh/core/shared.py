import numpy as np
import jax.numpy as jnp
import jax

def remap(value, low1, high1, low2, high2):
    '''Remap values from one domain to another'''
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def batch_data(data, batch_size):
    '''Break data into batches'''
    batches =[]
    for i in range(0, data.shape[0], batch_size):
        batches.append(data[i : i + batch_size, ...])
    return batches

def create_queries(width : int, height : int):
    '''Creates queries of specified resolution within normalized range -1.0 to 1.0.'''
    xs = np.linspace(-1, 1, width)
    ys = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(xs, ys)
    q = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    return q

def positional_encoding(x : jax.Array , num_frequencies : int):
    '''
    x: [..., D] input coordinates
    returns: [..., D * (1 + 2 * num_frequencies)]

    The returned encoding groups all encoded components for each input coordinate
    together. For each dimension d the output contains: [x_d, sin(2^0 x_d), cos(2^0 x_d),
    sin(2^1 x_d), cos(2^1 x_d), ...]. This makes the encoding coordinate-major
    (all components of x_d are contiguous) rather than frequency-major.

    NeRF commonly uses 10 frequencies per Cartesian coordinate and 4 for viewing direction.
    '''
    D = x.shape[-1]
    enc_per_dim = []
    for d in range(D):
        # work on a single coordinate column to keep its encoded components contiguous
        col = x[..., d:d+1]
        parts = []
        for i in range(num_frequencies):
            freq = 2.0 ** i
            parts.append(jnp.sin(freq * col))
            parts.append(jnp.cos(freq * col))
        enc_per_dim.append(jnp.concatenate(parts, axis=-1))
    return jnp.concatenate(enc_per_dim, axis=-1)