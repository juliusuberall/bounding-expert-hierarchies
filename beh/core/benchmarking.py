import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import time
import gc

from beh.registry import *

def get_fn_fp_rate(yp : np.ndarray, y : np.ndarray, threshold : float = 0.1):
    '''
    Measure conservativness of predictions and labels with False-Negative and False-Positive rate.
    '''
    fp_rate = np.sum((yp > threshold) * (y == 0)) / np.sum(y == 0)
    fn_rate = np.sum((yp < threshold) * (y > 0)) / np.sum(y > 0)
    return fn_rate, fp_rate

#------------------------------------------------------------------------------------

def benchmark_inference_speed(
    _ ,
    model,
    func,
    batch_size : int,
    iterations : int,
    query_size : int,
    query_dim : int,
    print_updates : bool = True):
    '''
    Measures the inference speed of a model given its forward call for N queries, median over M idedependant repitions.
    '''
    # Number of steps for n amount of total samples to be benchmarked per iteration
    total_steps = query_size // batch_size
    # Random key seed for every benchmarked model to ensure same randome queries
    master_key = jax.random.PRNGKey(28)

    @jax.jit
    def process_batched_queries(key):

        @jax.jit
        def batch_INF(i, carry):
            key, acc = carry
            # JAX's cheap and determenistic key splitter without dependency chain
            # Will give every benchmarked model the same random queries unless master PRNG key seed changed
            k = jax.random.fold_in(key, i)
            # Generate batch of random queries / Not possible to pre-compute due to memory limits
            x = jax.random.uniform(k, (batch_size, query_dim), minval=-1, maxval=1)
            # Forward queries through model
            y = func(model, x)
            # Use model output to ensur JIT does not eliminate dead-code
            acc = acc + jnp.sum(y) 
            return (key, acc)

        _ , acc = jax.lax.fori_loop(0, total_steps, batch_INF, (key, jnp.array(0.0)))
        return acc

    # Warm-up and JIT + check if GPU would run out of memory and skip otherwise
    try:
        _ = process_batched_queries(master_key).block_until_ready()
    except jaxlib.xla_extension.XlaRuntimeError as e:
        print(f"XLA Runtime Error - Device out of memory. Inference benchmark skipped.")
        # Ensure GPU memory is cleaned after OOM and return -1 as benchmark result
        jax.clear_caches()
        gc.collect()
        return jnp.array(-1.0)

    speed = []
    for i in range(iterations):
        # Measure speed and convert nano to miliseconds
        start = time.perf_counter_ns()
        # .block_until_ready() is essential here to measure the true computation time to account for JAX's asynchronous dispatch:
        # https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html#just-in-time-compilation-with-jax-jit
        _ = process_batched_queries(master_key).block_until_ready() 
        end = time.perf_counter_ns()
        speed.append((end - start) * 1e-6)

        if print_updates:
            end = '\n' if i == (iterations - 1) else '\r'
            print(f'{i+1}/{iterations} iterations with {int(query_size / 1e06)}M queries batched as {batch_size} -> {func.__name__}()', end = end, flush = True)

    # Median of speed measures from all iterations, to avoid outliers
    speed = jnp.median(jnp.array(speed))
    
    return speed