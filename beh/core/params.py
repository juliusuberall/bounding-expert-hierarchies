import jax
import jax.numpy as jnp


def init_layer(layer_dims, key):
    """
    Initalize parameters of network (weights and biases)

    Args
    ----------
    layer_dims :
        The layer dimensions. Including input, hidden and output layer dims.
    key :
        Jax random key used for sampling.
    """
    p = []
    keys = jax.random.split(key, len(layer_dims))
    for k, (n_in, n_out) in zip(keys, zip(layer_dims[:-1], layer_dims[1:])):
        # LeCun initialization
        w = jax.random.normal(k, (n_in + 1, n_out)) * jnp.sqrt(1 / n_in)
        # Zero all bias weights for legit initialization 
        w = w.at[n_in, ...].set(0)
        p.append(w)
    return p

#------------------------------------------------------------------------------------

def count_parameter(layer):
    """Count single networks parameter. Expects parameters to use bias trick"""
    p = 0
    for i in range(1, len(layer)):
        p += layer[i] * layer[i-1] + layer[i]
    return p

#------------------------------------------------------------------------------------

def init_moe(g_arch, e_arch, n_experts, key):
    """
    Initalize parameters of Mixture of Experts. Gate and expert parameters are stored seperate as list of arrays for each layer. 
    This allows to loop over all arrays in the list and select the corresponding expert by index.

    Args
    ----------
    g_arch :
      Gate architecture including input, hidden and output layer dims.
    e_arch :
      Expert architecture including input, hidden and output layer dims.
    n_experts:
      Number of experts to initalize.
    key :
      Jax random key used for sampling.
    
    Return
    ----------
    Dict :
      'gate': g, 'experts': e
    """
    # Init Gate parameters
    key, subkey = jax.random.split(key)
    g = []
    keys = jax.random.split(subkey, len(g_arch))
    for k, (n_in, n_out) in zip(keys, zip(g_arch[:-1], g_arch[1:])):
      # Account for bias by +1 input dim
      super_layer = jax.random.normal(k, (n_in + 1, n_out)) * jnp.sqrt(1 / n_in) # LeCun initialization
      # Zero all bias weights for legit initialization 
      super_layer = super_layer.at[n_in, ...].set(0)
      g.append(super_layer)

    # Init Experts parameters
    key, subkey = jax.random.split(key)
    e = []
    keys = jax.random.split(subkey, len(e_arch))
    for k, (n_in, n_out) in zip(keys, zip(e_arch[:-1], e_arch[1:])):
      # Account for bias by +1 input dim
      super_layer = jax.random.normal(k, (n_experts, n_in + 1, n_out)) * jnp.sqrt(1 / n_in) # LeCun initialization
      # Zero all bias weights for legit initialization 
      super_layer = super_layer.at[jnp.arange(n_experts), n_in, ...].set(0)
      e.append(super_layer)
    
    return {'gate': g, 'experts': e}