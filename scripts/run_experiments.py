import argparse
import jax

from beh.core.train_moe import train_moe
from beh.adapter.wiring import *

def main():

    key = jax.random.PRNGKey(28)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        type = str,
        default = None,
        help ='Name of training data')
    
    parser.add_argument(
        "--query",
        choices = [ "point"],
        type = str,
        default = None,
        help = "Query type") 
    
    parser.add_argument(
        "--dim",
        type = int,
        choices = [ 2, 3, 4, 10],
        default = None,
        help = "Query dimension")
    args = parser.parse_args()

    # Load data 
    data = get_traininig_data()

    # Instantiate registry for storing results of these experiments 

    # Train models 
    train_moe(
        key = key,
        training_data = data,
        query = args.query,
        dimension = args.dim)

    # Save results to file in the 'results' directory at the project root


if __name__ == '__main__':
    main()