from beh.adapter.sample_dim3 import *
from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *
from beh.core.registry import *

from beh.adapter.sample import load_samples

def get_traininig_data(
    data_name : str , 
    dimension : int ,
    reg : CoreRegistry
    ):
    '''
    Load and extract training data.
    \nDelegation based on dimensionalty to ensure correct data extraction procedure.
    '''

    if dimension == 2:
        return preprocess_rgba(data_dir_registry[dimension] + f"/{data_name}.png", reg)
    elif dimension == 3:
        return load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg)
    elif dimension == 4:
        pass
    elif dimension == 10:
        pass
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
def sample(dimension : int , args):
    '''
    Sample data.
    \nDelegation based on dimensionalty to ensure correct data sampling procedure.
    '''
    if args.strategy == 'grid':
        if dimension == 2:
            pass
        elif dimension == 3:
            return sample_obj_grid(args)
        elif dimension == 4:
            pass
        elif dimension == 10:
            pass
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
    else:
        if dimension == 2:
            pass
        elif dimension == 3:
            return sample_obj_random(args)
        elif dimension == 4:
            pass
        elif dimension == 10:
            pass
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")