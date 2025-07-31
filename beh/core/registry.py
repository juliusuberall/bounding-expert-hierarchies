# Class to register all relevant experiment data
class CoreRegistry:
    def __init__(self):
        self.registry = {}

    def get(self, name : str):
        if name in self.registry.keys():
            return self.registry[name]
        else:
            raise ValueError("metric {} not registered".format(name))

    def add(self, name : str, value):
        self.registry[name] = value

# Data directory managment
core_keys = {
    'sparse_fn_key' : '_sparse_fn', # False Negative rate in %
    'sparse_fp_key' : '_sparse_fp', # False Positive rate in %
    'accuracy_mse_key' : '_accuracy_mse', # Stores the mean squared error of the accuracy benchmark
    'expert_boundary_key' : '_expert_boundary', # The individual full decision boundaries for all experts
    'gate_top1_activation_key' : '_top1_activation', # Map of top1 activation indicies of experts
    'y_prediciton_key' : '_yp', # All remapped outputs predicited during accuracy compute
    'y_prediciton_RAW_key' : '_yp_raw', # All NOT REMAPPED outputs predicited during accuracy compute
    'data_resolution_key' : 'data_res', # Stores data resolution, capturing width and height, depth etc.
    'gating_confidence_key' : '_gate_confidence', # The sparsity of the MoE gate
    'gating_sorted_activation_key' : '_gate_sorted_activation', # The gate activation probabilities sorted from max to min
    'train_val_loss_key' : '_training_validation_loss', # Stores training validation loss based on logging frequency in config
    'train_confidence_key' : '_training_confidence', # Gating confidence logged throughout training
    'train_epoch_key' : '_training_epoch', # Epoch timestamp for logged training metric following training logging frequency from config
}