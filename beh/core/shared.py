import jax

def remap(value, low1, high1, low2, high2):
    # Remap values from one domain to another
    ## Taken from: https://stackoverflow.com/questions/3451553/value-remapping
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def batch_data(data, batch_size):
    # Break data into batches
    batches =[]
    for i in range(0, data.shape[0], batch_size):
        batches.append(data[i : i + batch_size, ...])
    return batches