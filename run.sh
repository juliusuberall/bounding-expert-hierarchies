# Add the root directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)


# 2D - Experiments
## Define 2D bounding objects used for experiments
objects=("dolphin" "freeform")

## Loop over objects
for object in "${objects[@]}"; do

    # Define command line arguments for query
    point="--data_name ${object} --query point --dim 2"

    # Execute python script with arguments
    python3 scripts/run_experiments.py $point

done


# 3D - Experiments


# 4D - Experiments


# 4D+ - Experiments


# Create result plots and figures
