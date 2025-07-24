import jax

def batch_data(data : jax.Array, batch_size):
    # Break data into batches
    batches =[]
    for i in range(0, data.shape[0], batch_size):
        batches.append(data[i : i + batch_size, ...])
    return batches