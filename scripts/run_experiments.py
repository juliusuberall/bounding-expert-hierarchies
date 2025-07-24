import argparse
import yaml
import jax

from beh.core.train_moe import train_moe
from beh.adapter.wiring import *
from beh.styler.wiring import *

def main():

    key = jax.random.PRNGKey(28)

    # Check that all registry paths and folder exists to catch errors before code runs

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        required=True, 
        type = str,
        default = None,
        help ='Name of training data')
    
    parser.add_argument(
        "--query",
        required=True, 
        choices = ["point"],
        type = str,
        default = None,
        help = "Query type") 
    
    parser.add_argument(
        "--dim",
        required=True, 
        type = int,
        choices = [ 2, 3, 4, 10],
        default = None,
        help = "Query dimension")
    args = parser.parse_args()

    # Load, preprocess and check data once before trained with all models
    x, y = get_traininig_data(
        data_name = args.data_name,
        dimension = args.dim
    )
    checkpoint_training_data(x, y)
    checkpoint_plot_training_data(x, y, args.dim)

    # Load model configurations / architectures
    with open(model_config_dir_registry[args.dim], "r") as file:
        configs = yaml.safe_load(file)

    # Instantiate registry for storing results of these experiments 

    # Train models 
    moe = train_moe(
        key = key,
        x = x,
        y = y,
        configs = configs,
        query = args.query,
        dimension = args.dim)

    # Save results to file in the 'results' directory at the project root


if __name__ == '__main__':
    main()