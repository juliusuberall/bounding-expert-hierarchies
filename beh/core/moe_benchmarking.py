import jax
import jax.numpy as jnp
import optax

from beh.core.shared import *
from beh.core.moe import *

def accuracy_benchmark(
        moe : dict,
        x : jax.Array, 
        y : jax.Array,
        configs,
    ):

    # Extract model configurations
    batch_size = configs['general']['batch_size']
    
    # Batch data
    x_batches = batch_data(x, batch_size)

    # Dense MoE inference
    dense_mse = moe_loss_dense(x_batches, y, moe)
    dense_mse = round(float(dense_mse),4)

    # Sparse MoE inference
    sparse_mse = moe_loss_sparse(x_batches, y, moe)
    sparse_mse = round(float(sparse_mse),4)


    print(f"Dense Prediction MSE: {dense_mse}")
    print(f"Sparse Prediction MSE: {sparse_mse}")