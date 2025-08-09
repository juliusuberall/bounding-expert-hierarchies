import argparse
import yaml

from beh.registry import *

def parse_train():
    '''
    Retrieve command line arguments and model configuration yaml for invoking model training and evaluation.
    '''
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
        choices = [ 2, 3, 4, 9],
        default = None,
        help = "Query dimension")
    args = parser.parse_args()

    # Load model configurations / architectures
    with open(model_config_registry[args.dim], "r") as file:
        configs = yaml.safe_load(file)

    return args, configs

def parse_sample():
    '''
    Retrieve command line arguments for invoking data sampling.
    '''
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        required=True, 
        type = str,
        default = None,
        help ='Name of data to sample')
    parser.add_argument(
        "--dim",
        required=True, 
        type = int,
        choices = [ 2, 3, 4, 9],
        default = None,
        help = "Query dimension")
    parser.add_argument(
        "--res",
        required=False, 
        type = int,
        default = None,
        help = "Dimension Resolution")
    parser.add_argument(
        "--strategy",
        required=False, 
        type = str,
        choices = [ 'random', 'grid'],
        default = None,
        help = "Sampling Strategy")
    
    return parser.parse_args()