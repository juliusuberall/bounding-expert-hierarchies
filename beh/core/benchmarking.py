import jax
import jax.numpy as jnp
import numpy as np
import time

def get_fn_fp_rate(yp : jax.Array, y : jax.Array, threshold : float = 0.1):
    '''
    Measure conservativness of predictions and labels with False-Negative and False-Positive rate.
    '''
    fp_rate = jnp.sum((yp > threshold) * ( y == 0)) / y.size
    fn_rate = jnp.sum((yp < threshold) * ( y > 0)) / y.size
    return fn_rate, fp_rate

def min_inference_speed(
        x_batches : list,
        model, 
        func, 
        iterations : int,
        iteration_size : int,
        configs : dict,
        dimension : int):
    '''
    Measures the minimum inference speed of a MoE for N batched queries, averaged over M iterations.
    \nRandomly creates batches from the data provided to this function.
    '''
    # General batch size is defined for all models of same dimensionalty 
    batch_size = configs['general']['batch_size']

    # General setup
    x = jnp.stack(x_batches[:-1]).reshape((-1,dimension))
    min_speed = []

    # Warm-up to ensure JAX JIT compilation, GPU caches
    warm_it = 100
    for i in range(warm_it):
        xj = x[np.random.choice(x.shape[0], batch_size)]
        jax.vmap(lambda x: func(model, x))(xj).block_until_ready()
        end = '\n' if i == (iterations - 1) else '\r'
        print(f'{i+1}/{warm_it} WARM-UP iterations of batch size {batch_size} -> {func.__name__}()', end = end, flush = True)

    for i in range(iterations):
        speed = []
        for j in range(iteration_size):
            xj = x[np.random.choice(x.shape[0], batch_size)]

            # Measure inference
            start = time.perf_counter()
            jax.vmap(lambda x: func(model, x))(xj).block_until_ready()
            end = time.perf_counter()

            speed.append((end - start) * 1e6) # Convert nano to miliseconds
        min_speed.append(np.min(speed))
        end = '\n' if i == (iterations - 1) else '\r'
        print(f'{i+1}/{iterations} iterations with {iteration_size} repeated queries of batch size {batch_size} -> {func.__name__}()', end = end, flush = True)
    
    # Averages the minimum measures from all iterations
    min_speed = jnp.mean(jnp.array(min_speed))

    return min_speed