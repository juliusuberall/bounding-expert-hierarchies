import jax
import beh.registry

from beh.adapter.wiring import *
from beh.styler.wiring import *
from beh.core.registry import *
from beh.core.moe_benchmarking import *
from beh.config_parser import *
from beh.core.wiring import *
from beh.shared import setup_dirs

from beh.styler.shared import export_plot_model_comparison_graph

def main():

    key = jax.random.PRNGKey(28)
    setup_dirs(beh.registry)
    args, configs = parse_train()
    print(f"\n███████████████████████{args.dim}D█████████████████████████")

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

    # Loop over all configured models
    ## Train, Benchmark, Save model
    for i in configs.items():
        
        # Get model name
        model_key = i[0]
        if model_key == 'general': continue
        print(f"\n================================================== {model_key}")
        
        model, reg = train_model(
            model_key = model_key,
            key = key,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            query = args.query,  # Currently not used but ideally for point and ray queries
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
            model_key = model_key,
            x = x,
            y = y,
            reg = reg,
            configs = configs,
            dimension = args.dim,
            data_name = args.data_name
        )

        # save_model(
        #     model_key = model_key,
        #     configs = configs,
        #     model = model
        # )

    export_plot_model_comparison_graph(
        reg = reg,
        configs = configs,
        dimension = args.dim)

if __name__ == '__main__':
    main()