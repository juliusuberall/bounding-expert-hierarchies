import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import time
import gc
import matplotlib.pyplot as plt

from beh.registry import *

def get_fn_fp_rate(yp : jax.Array, y : jax.Array, threshold : float = 0.1):
    '''
    Measure conservativness of predictions and labels with False-Negative and False-Positive rate.
    '''
    fp_rate = jnp.sum((yp > threshold) * (y == 0)) / jnp.sum(y == 0)
    fn_rate = jnp.sum((yp < threshold) * (y > 0)) / jnp.sum(y > 0)
    return fn_rate, fp_rate

#------------------------------------------------------------------------------------

def inference_speed(
    x,
    model,
    func,
    reps : int,
    query_size : int,
    dimension : int,
    inf_batch_size : int,
    sparse : bool = False):
    '''
    Measures the inference speed of a model with an ideal batch size for max. speed.
    '''

    # Create queries and upload once to device
    ## Replace true makes this much faster allowing for duplicates
    idx = np.random.choice(np.arange(x.shape[0]), query_size, replace=True)
    x = x[idx,...]

    # Benchmark
    speed = benchmark_inference_speed(
        x=x,
        model=model,
        func=func,
        batch_size=inf_batch_size,
        reps=reps,
        query_size=query_size,
        dimension=dimension
    )
    return speed

#------------------------------------------------------------------------------------

def benchmark_inference_speed(
    x,
    model,
    func,
    batch_size : int,
    reps : int,
    query_size : int,
    dimension : int,
    print_updates : bool = True):
    '''
    Measures the inference speed of a model given its forward call for N queries, median over M idedependant repitions.
    '''
    @jax.jit
    def batch_INF(i, val):
        start = i * batch_size
        # Dynamic slicing allows us to sub-select on the device, avoiding transfer overhead.
        # Dynamic slice handles tail if query-size and batch-size are not % == 0. In that
        # case it moves the start index as required.
        x_slice = jax.lax.dynamic_slice(x, (start, 0), (batch_size, dimension))
        y = func(model, x_slice)
        return val + jnp.sum(y) # Passing to make sure JIT does not ignore forward call

    # Number of steps needed to go through entire query using defined batch size
    # Ensure loop hits tail, by always rounding up
    full_batch_iter = query_size // batch_size + bool(query_size % batch_size)

    @jax.jit
    def fori():
        return jax.lax.fori_loop(0, full_batch_iter, batch_INF, 0.0)

    # Warm-up and JIT compile
    _ = fori().block_until_ready()

    speed = []
    for i in range(reps):
        # Measure speed and convert nano to miliseconds
        start = time.perf_counter_ns()
        _ = fori().block_until_ready()
        end = time.perf_counter_ns()
        #print("-> One time sample measure completed")
        speed.append((end - start) * 1e-6)

        if print_updates:
            end = '\n' if i == (reps - 1) else '\r'
            print(f'{i+1}/{reps} iterations with {int(query_size / 1e06)}M queries batched as {batch_size} -> {func.__name__}()', end = end, flush = True)

    # Median of speed measures from all iterations, to avoid outliers
    speed = jnp.median(jnp.array(speed))

    return speed