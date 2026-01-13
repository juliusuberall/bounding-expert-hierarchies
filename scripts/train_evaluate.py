import jax
import beh.registry
import time

from beh.adapter.wiring import *
from beh.styler.wiring import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.config_parser import *
from beh.core.wiring import *
from beh.shared import setup_dirs, move_results
from beh.registry import *

def main():

    key = jax.random.PRNGKey(28)
    setup_dirs(beh.registry)
    args, configs = parse_train()
    print(f"\n███████████████████████{args.dim}D█████████████████████████")

    # Instantiate registry for storing results of these experiments 
    reg = CoreRegistry()

    # Load, preprocess and check data once before trained with all models
    reg, x, y = get_traininig_data(
        query = args.query,
        data_name = args.data_name,
        dimension = args.dim,
        reg = reg
    )
    checkpoint_training_data(x, y)

    # Loop over all configured models
    ## Train, Benchmark, Save model
    for i in configs.items():
        
        # Get model name
        model_key = i[0]
        if model_key == 'general': continue
        reg.add(model_key + core_keys['experiment_start_time_key'], time.perf_counter_ns())
        print(f"\n================================================== {model_key}")
        
        model, reg = train_model(
            model_key = model_key,
            key = key,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            query_dim = x.shape[-1], 
            dimension = args.dim
        )
    
        reg = get_benchmarks(
            model_key = model_key,
            model = model,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            dimension = args.dim
        )
    
        format_export_results(
            model = model,
            model_key = model_key,
            y = y,
            reg = reg,
            configs = configs,
            dimension = args.dim,
            data_name = args.data_name,
            query = args.query
        )
    
    # Bundle all exported results by data-name prefix
    move_results(result_dir_registry[args.dim], args.data_name)

if __name__ == '__main__':
    main()