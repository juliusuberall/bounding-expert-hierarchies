from beh.adapter.dim3 import *
from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *
from beh.core.registry import *

import beh.adapter.dim3 as dim3
import beh.adapter.dim4 as dim4
import beh.adapter.dim4plus as dim4plus

def get_traininig_data(
    data_name : str , 
    dimension : int ,
    reg : CoreRegistry = CoreRegistry()
    ):
    '''
    Load and extract training data.
    \nDelegation based on dimensionalty to ensure correct data extraction procedure.
    '''
    if dimension == 2:
        return preprocess_rgba(data_dir_registry[dimension] + f"/{data_name}.png", reg)
    elif dimension == 3:
        return dim3.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg)
    elif dimension == 4:
        return dim4.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg)
    elif dimension == 9:
        return dim4plus.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg)
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
def pre_process(dimension : int , args):
    '''
    Sample data.
    \nDelegation based on dimensionalty to ensure correct data sampling procedure.
    '''
    if dimension == 3:
        if args.strategy == 'grid':
            return sample_obj_grid(args)
        else:
            return sample_obj_random(args)
    elif dimension == 4:
        return dim4.preprocess_4D(args)
    elif dimension == 9:
        return dim4plus.preprocess_4Dplus(args)
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")