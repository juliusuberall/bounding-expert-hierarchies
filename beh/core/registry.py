import jax
import numpy as np

# Class to register all relevant experiment results and data
class CoreRegistry:
    def __init__(self):
        self.registry = {}

    def get(self, name : str):
        if name in self.registry.keys():
            return self.registry[name]
        else:
            raise ValueError(f"Metric {name} not registered")

    def add(self, name : str, value):
        # Ensure we only store numpy arrays and not jax arrays which could be on device
        if isinstance(value, jax.Array):
            value = np.array(value)
        self.registry[name] = value

# Data directory managment
core_keys = {
    'aa_y_prediciton_key' : 'aa_y',                               # For visulaizing the binary classification in 2D we query in higher resolution and downsample to original for anti-aliasing
    'aa_gate_top1_activation_key' : '_top1_activation_aa',        # Map of top1 activation indicies of experts for anti-aliasing high res queries
    'architecture' : '_model_arch',                               # String formatted model architecture details, layer numbers and size
    'active_parameters_key' : '_act_para',                        # Total parameters
    'active_experts_key' : '_act_e',                              # Cache of active experts throughout training 
    'data_size_key' : 'data_size_per_dim',                        # Stores data size, capturing width and height, depth etc.
    'data_bounds_key' : 'data_bounds',                            # Stores data original data bounds so we can translate back correct after normalization
    'experiment_start_time_key' : '_exp_start_time',              # Timestamp of experiment start e.g. for measuring total epxeriment time of individual models incl. benchmarking
    'fn_key' : '_fn',                                             # False Negative rate in %
    'fp_key' : '_fp',                                             # False Positive rate in %
    'gate_top1_activation_key' : '_top1_activation',              # Map of top1 activation indicies of experts
    'gating_confidence_key' : '_gate_confidence',                 # The sparsity of the MoE gate
    'gating_sorted_activation_key' : '_gate_sorted_activation',   # The gate activation probabilities sorted from max to min
    'inf_speed_key' : '_inf_speed',                               # List of pairs: batchsize + inference speed in miliseconds
    'train_confidence_key' : '_training_confidence',              # Gating confidence logged throughout training
    'train_epoch_key' : '_training_epoch',                        # Epoch timestamp for logged training metric following training logging frequency from configs
    'train_fn_key' : '_training_fn',                              # False negative rate training cache
    'train_fp_key' : '_training_fp',                              # False negative rate training cache
    'training_time' : '_train_time',                              # Training time in s
    'train_conservative_experts_key' : 'con_e',                   # Conservative experts in each training iteration
    'total_epochs' : "total_trained_epochs",                      # The number of epochs until training stopped
    'total_parameters_key' : '_tot_para',                         # Total parameters
    'y_prediciton_key' : '_yp',                                   # All remapped outputs predicited during accuracy compute
}

# Space partitioning routing table
# As we are gating in MoEG and MOE based on positional location for point-queries and ray-queries
# we define a table for looking up to which index the location is encoded for varying query dimensions.
gating_table = {
    'point' : {
        2 : 2, # 2D point
        3 : 3, # 3D point
        4 : 4, # 4D point
        9 : 9, # 9D control state space - (3D position in space; 6D robot joints for pose)
    },
    'ray' : { # We only route based on location 
        4 : 2, # 2D ray (2D point + 2D ray)
        6 : 3, # 3D ray (3D point + 3D ray)
        7 : 4, # 4D ray (4D point + 3D ray)
    }
} 