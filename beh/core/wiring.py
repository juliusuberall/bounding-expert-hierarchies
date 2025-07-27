from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *

# An orchestration function that loads the correct data 
def get_benchmarks(
        moe : dict,
        x : jax.Array,
        y : jax.Array,
        reg : CoreRegistry,
        configs : dict,
        dimension : int
    ):

    # Batch data
    batch_size = configs['general']['batch_size']
    x_batches = batch_data(x, batch_size)

    if dimension == 2:
        reg = register_accuracy(moe, x_batches, y, reg)
        reg = register_gating_confidence(moe, x_batches, reg)
        reg = register_all_expert_boundaries(moe, x_batches, reg)
        return reg
    
    elif dimension == 3:
        pass
    elif dimension == 4:
        pass
    elif dimension == 10:
        pass
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")