import argparse
import yaml
import jax

from beh.core.train_moe import train_moe
from beh.adapter.wiring import *
from beh.styler.wiring import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.config_parser import *

def main():

    key = jax.random.PRNGKey(28)

    # Check that all registry paths and folder exists to catch errors before code runs
    args, configs = parse_config()

    # Load, preprocess and check data once before trained with all models
    x, y = get_traininig_data(
        data_name = args.data_name,
        dimension = args.dim
    )
    checkpoint_training_data(x, y)
    checkpoint_plot_training_data(x, y, args.dim)

    # Instantiate registry for storing results of these experiments 
    reg = CoreRegistry()

    # MoE
    ## Training 
    moe, reg = train_moe(
        key = key,
        x = x,
        y = y,
        reg = reg,
        configs = configs,
        query = args.query,  # Currently not used but ideally for point and ray queries
        dimension = args.dim)
    
    ## Benchmarking
    reg = accuracy_benchmark(
        moe = moe,
        x = x,
        y = y,
        configs = configs
    )
    
    ## Result Collection
    # generate_model_output(
    #     model = moe,
    #     x = x,
    #     configs = configs,
    #     dimension = args.dim)
    


    ## MLP


    # Save results to file in the 'results' directory at the project root


if __name__ == '__main__':
    main()