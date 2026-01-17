from beh.adapter.dim2 import *
from beh.adapter.shared import *
from beh.registry import *
from beh.core.registry import *

import beh.adapter.dim2 as dim2
import beh.adapter.dim3 as dim3
import beh.adapter.dim4 as dim4
import beh.adapter.dim4plus as dim4plus

def get_traininig_data(
    query : str,
    data_name : str , 
    dimension : int ,
    reg : CoreRegistry = CoreRegistry()
    ):
    '''
    Load and extract training data.
    \nDelegation based on dimensionalty to ensure correct data extraction procedure.
    '''
    if dimension == 2:
        return dim2.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg, query, dimension, data_name)
    elif dimension == 3:
        return dim3.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg, query)
    elif dimension == 4:
        return dim4.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg, query)
    elif dimension == 9:
        return dim4plus.load_samples(data_dir_registry[dimension] + f"/{data_name}.npz", reg)
    else:
        raise ValueError(f"Unsupported data dimensionality: {dimension}")
    
def pre_process(args):
    '''
    Sample data.
    \nDelegation based on dimensionalty to ensure correct data sampling procedure.
    '''
    dimension = args.dim
    if args.query == 'point':
        if dimension == 3:
            if args.strategy == 'grid':
                return dim3.sample_pts_grid(args)
            else:
                return dim3.sample_pts_random(args)
        elif dimension == 4:
            return dim4.preprocess_4D_points(args)
        elif dimension == 9:
            return dim4plus.preprocess_4Dplus(args)
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")
    elif args.query == 'ray':
        if dimension == 2:
            return dim2.sample_rays_random(args)
        elif dimension == 3:
            return dim3.sample_rays_random(args)
        elif dimension == 4:
            return dim4.sample_rays_random(args)
        elif dimension == 9:
            pass
        else:
            raise ValueError(f"Unsupported data dimensionality: {dimension}")