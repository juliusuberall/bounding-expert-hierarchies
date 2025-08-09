# Data directory managment
data_dir_registry = {
    2 : "./data/2D", 
    3 : "./data/3D", 
    4 : "./data/4D", 
    9 : "./data/4D_plus", 
}

# Model configs path managment
model_config_registry = {
    2 : "./configs/models-2D.yaml", 
    3 : "./configs/models-3D.yaml", 
    4 : "./configs/models-4D.yaml", 
    9 : "./configs/models-4Dplus.yaml",
}

# Result data directory managment 
result_dir_registry = {
    2 : "./results/2D", 
    3 : "./results/3D", 
    4 : "./results/4D",
    9 : "./results/4D_plus",
}

# Model directory managment 
result_model_dir_registry = {
    2 : result_dir_registry[2] + "/models", 
    3 : result_dir_registry[3] + "/models", 
    4 : result_dir_registry[4] + "/models", 
    9 : result_dir_registry[9] + "/models", 
}