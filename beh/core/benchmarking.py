import jax
import jax.numpy as jnp
import numpy as np
import time
import timeit

def get_fn_fp_rate(yp : jax.Array, y : jax.Array, threshold : float = 0.1):
    '''
    Measure conservativness of predictions and labels with False-Negative and False-Positive rate.
    '''
    fp_rate = jnp.sum((yp > threshold) * ( y == 0)) / y.size
    fn_rate = jnp.sum((yp < threshold) * ( y > 0)) / y.size
    return fn_rate, fp_rate

#------------------------------------------------------------------------------------

def min_inference_speed(
        x : jax.Array,
        model, 
        func, 
        batch_size : int,
        reps : int,
        query_size : int,
        dimension : int):
    '''
    Measures the inference speed of a MoE for N queries, averaged over M iterations.
    '''
    # Create queries
    x = jax.device_put(x).block_until_ready()
    idx = np.random.choice(np.arange(x.shape[0]), query_size, replace=True) # Replace true makes this much faster
    x = x[idx,...]
    full_batch_iter = query_size // batch_size

    @jax.jit
    def batch_INF(i, val):
        start = i * batch_size
        x_slice = jax.lax.dynamic_slice(x, (start, 0), (batch_size, dimension))
        y = jax.vmap(lambda x: func(model, x))(x_slice)
        return val + jnp.sum(y)

    @jax.jit
    def fori():
        return jax.lax.fori_loop(1, full_batch_iter, batch_INF, 0.0)

    # Warm-up and JIT compile
    _ = fori().block_until_ready()
    
    speed = []
    for i in range(reps):
        # Measure speed and convert nano to miliseconds
        start = time.perf_counter_ns()
        _ = fori().block_until_ready()
        end = time.perf_counter_ns()
        speed.append((end - start) * 1e-6)

        end = '\n' if i == (reps - 1) else '\r'
        print(f'{i+1}/{reps} iterations with {query_size} queries batched as {batch_size} -> {func.__name__}()', end = end, flush = True)
    
    # Averages the speed measures from all iterations
    speed = jnp.mean(jnp.array(speed))

    return speed