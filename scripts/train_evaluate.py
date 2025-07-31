import argparse
import yaml
import jax

from beh.core.train_moe import train_moe
from beh.core.train_mlp import train_mlp
from beh.adapter.wiring import *
from beh.styler.wiring import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.config_parser import *
from beh.core.wiring import *

def main():

    key = jax.random.PRNGKey(28)

    # Check that all registry paths and folder exists to catch errors before code runs
    args, configs = parse_all()

    # Instantiate registry for storing results of these experiments 
    reg = CoreRegistry()

    # Load, preprocess and check data once before trained with all models
    reg, x, y = get_traininig_data(
        data_name = args.data_name,
        dimension = args.dim,
        reg = reg
    )
    checkpoint_training_data(x, y)
    checkpoint_plot_training_data(x, y, args.dim)

    # MLP
    ## Training
    mlp, reg = train_mlp(
        key = key,
        x = x,
        y = y,
        reg = reg,
        configs = configs,
        query = args.query,  # Currently not used but ideally for point and ray queries
        dimension = args.dim
    )

    # MoE
    ## Training 
    moe, reg = train_moe(
        key = key,
        x = x,
        y = y,
        reg = reg,
        configs = configs,
        query = args.query,  # Currently not used but ideally for point and ray queries
        dimension = args.dim
    )
    
    ## Benchmarking
    #### Loop over all models and pass model name for sving and loading to correct result key
    reg = get_benchmarks(
        moe = moe,
        x = x,
        y = y,
        reg = reg,
        configs = configs,
        dimension = args.dim
    )
    
    # Result Collection
    format_export_results(
        x = x,
        y = y,
        reg = reg,
        configs = configs,
        dimension = args.dim,
    )

    # Save models
    save_model(
        model = moe
    )


if __name__ == '__main__':
    main()