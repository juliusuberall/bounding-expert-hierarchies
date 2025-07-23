from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *

# An orchestration function that loads the correct data 
def get_traininig_data(
    data_name : str , 
    dimension : int ):

    if dimension == 2:
        return preprocess_rgba(data_dir_registry[dimension] + f"/{data_name}.png")
    elif dimension == 3:
        pass
    elif dimension == 4:
        pass
    elif dimension == 10:
        pass
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")