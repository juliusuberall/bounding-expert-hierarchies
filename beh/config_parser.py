import argparse
import yaml

from beh.registry import *

def parse_config():
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

    # Load model configurations / architectures
    with open(model_config_dir_registry[args.dim], "r") as file:
        configs = yaml.safe_load(file)

    return args, configs