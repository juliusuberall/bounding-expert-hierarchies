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
    model_key : str,
    sparse : bool = False):
    '''
    Measures the inference speed of a model with an ideal batch size for max. speed.
    '''

    # Create queries and upload once to device
    idx = np.random.choice(np.arange(x.shape[0]), query_size, replace=True) # Replace true makes this much faster
    x = x[idx,...]
    x = jax.device_put(x).block_until_ready()

    # Find optimal batch size for device and FLOPs cause by single query
    batch_size = get_optimal_batch_size(
        x=x,
        model=model,
        func=func,
        query_size=query_size,
        dimension=dimension,
        model_key=model_key,
        sparse=sparse
    )

    # Benchmark
    print(f"Optimal batch size => {batch_size}")
    speed = benchmark_inference_speed(
        x=x, 
        model=model,
        func=func,
        batch_size=batch_size,
        reps=reps,
        query_size=query_size,
        dimension=dimension
    )
    return speed, batch_size

#------------------------------------------------------------------------------------

def get_optimal_batch_size(
    x,
    model,
    func,
    query_size : int,
    dimension : int,
    model_key : str,
    sparse : bool = False):
    '''
    Find the optimal batch size for model and device, such that we can forward 
    through the function the fastest for a large number of queries. 
    \nExport plot of batch size - inference speed trend.
    '''
    speed = []
    start_batch_size, optimal_batchsize = 1024, 0
    max = 13
    print(f"\nFinding optimal batch size")
    for i in range(0, max):
        try:
            speed.append(benchmark_inference_speed(
                x,
                model,
                func,
                start_batch_size * 2**i,
                10,
                query_size,
                dimension,
                False 
            ))
            if jnp.abs(speed[-1] - jnp.mean(jnp.array(speed[-4:-1]))) < speed[-1] * 0.05:
                optimal_batchsize = start_batch_size * 2**(i - 1)
                print(f"Inference speed plateaus.")
                break

        except jaxlib.xla_extension.XlaRuntimeError as e:
            optimal_batchsize = start_batch_size * 2**(i - 1)
            print(f"XLA Runtime Error - Device out of memory.")
            break

        finally:
            # Ensure GPU memory is cleaned such that we can actually find 
            # the largest working batchsize, since each time the shape changes
            # a new kernel is compiled and steals memory space
            jax.clear_caches()
            gc.collect()
        print(f'{i}/{max} tested batch size {int(start_batch_size * 2**i)} on {int(query_size / 1e06)}M queries -> mlp_forward_INF() | Inf. Speed: {speed[-1]}ms')

    # In case OOM never reached on device
    if optimal_batchsize == 0:
        optimal_batchsize = start_batch_size * 2**i

    # Final validation that final batch size was actually fastest measure
    speed_jax = jnp.array(speed)
    if speed[-1] > jnp.min(speed_jax):
        idx = jnp.argmin(speed_jax)
        optimal_batchsize = start_batch_size * 2**idx

    # Create and export plot
    plt.plot(np.arange(len(speed)), np.array(speed))
    plt.title(f"Batch Size - Inference Speed relation for {model_key}\nBest: {optimal_batchsize} -> {round(float(speed[-1]),2)}ms")
    plt.xlabel(f'Batch Size = {start_batch_size} * 2**x')
    plt.ylabel(f"Speed")
    if sparse:
        path = result_dir_registry[dimension] + f"/{model_key}_sparse_{dimension}D_optimal_batchsize.png"
    else:
        path = result_dir_registry[dimension] + f"/{model_key}_{dimension}D_optimal_batchsize.png"
    plt.savefig(path)
    plt.close()

    return optimal_batchsize

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
        speed.append((end - start) * 1e-6)

        if print_updates:
            end = '\n' if i == (reps - 1) else '\r'
            print(f'{i+1}/{reps} iterations with {int(query_size / 1e06)}M queries batched as {batch_size} -> {func.__name__}()', end = end, flush = True)
    
    # Median of speed measures from all iterations, to avoid outliers
    speed = jnp.median(jnp.array(speed))

    return speed