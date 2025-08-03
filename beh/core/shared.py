import jax
import time
import jax.numpy as jnp
import numpy as np

def remap(value, low1, high1, low2, high2):
    # Remap values from one domain to another
    ## Taken from: https://stackoverflow.com/questions/3451553/value-remapping
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def batch_data(data : jax.Array, batch_size):
    # Break data into batches
    batches =[]
    for i in range(0, data.shape[0], batch_size):
        batches.append(data[i : i + batch_size, ...])
    return batches

#------------------------------------------------------------------------------------

def min_inference_speed(
        x_batches : jax.Array,
        moe : dict, 
        func, 
        iterations : int,
        iteration_size : int,
        configs : dict):
    '''
    Measures the minimum inference speed of a MoE for N batched queries, averaged over M iterations.
    \nRandomly creates batches from the data provided to this function.
    '''
    # Generla batch size is defined for all models of same dimensionalty 
    batch_size = configs['general']['batch_size']

    x = jnp.array(x_batches[:-1]).reshape((-1,2))
    min_speed = []
    for i in range(iterations):
        speed = []
        for j in range(iteration_size):
            xj = x[np.random.choice(x.shape[0], batch_size)]

            # Measure inference
            start = time.perf_counter()
            jax.vmap(lambda x: func(moe, x))(xj).block_until_ready()
            end = time.perf_counter()

            speed.append((end - start) * 1e6)  # convert to microseconds
        min_speed.append(np.min(speed))
        end = '\n' if i == (iterations - 1) else '\r'
        print(f'{i+1}/{iterations} iterations with {iteration_size} repeated queries of batch size {batch_size} -> {func.__name__}()', end = end, flush = True)
    
    # Averages the minimum measures from all iterations
    min_speed = jnp.mean(jnp.array(min_speed))

    return min_speed