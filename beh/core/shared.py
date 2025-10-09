import numpy as np

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

def create_queries(width : int, height : int):
    '''Creates queries of specified resolution within normalized range -1.0 to 1.0.'''
    xs = np.linspace(-1, 1, width)
    ys = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(xs, ys)
    q = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    return q