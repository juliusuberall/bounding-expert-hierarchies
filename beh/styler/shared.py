import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from beh.registry import *
from beh.core.registry import *

# Plot the gradient of a neural network training
def checkpoint_moe_export_plot_gradient(gradient, dimension, epoch):
    '''Visualize the gradient during training and export as plot.'''
  
    # Create plot
    plt.title(f"Learning Signal {dimension}D MoE | Epoch {epoch}")
    for l in gradient['gate']:
        plt.plot(l.flatten(), linewidth=0.5)
        v, i = jax.lax.top_k(l.flatten(), 1)
        plt.scatter(i,v,label='GL')
    for l in gradient['experts']:
        plt.plot(l.flatten(), linewidth=0.5)
        v, i = jax.lax.top_k(l.flatten(), 1)
        plt.scatter(i, v, label='EL')
    plt.xlabel('Weights')
    plt.ylabel('Signal Strength')
    plt.legend()

    # Export
    path = result_dir_registry[dimension] + f"/training_gradient.png"
    plt.savefig(path)
    plt.close()
    pass

def export_plot_training_metrics (
    reg : CoreRegistry,
    dimension : int):
    '''
    Create a training metrics plot, showing the trend throughout training inlcuding:
    \n- Loss
    '''

    # Retrieve model specific key for results
    model_key = 'moe'

    # Get training metrics from registry
    loss = reg.get(model_key + core_keys['train_val_loss_key'])
    confidence = reg.get(model_key + core_keys['train_confidence_key'])
    fn = reg.get(model_key + core_keys['train_fn_key'])
    fp = reg.get(model_key + core_keys['train_fp_key'])
    epochs = reg.get(model_key + core_keys['train_epoch_key'])

    # Create plot 
    plt.title(f'{model_key} training')
    plt.plot(epochs, loss, label='Loss')
    plt.plot(epochs, confidence, label='Gate Confidence')
    plt.plot(epochs, fn, label='False Negatives')
    plt.plot(epochs, fp, label='False Positives')
    plt.legend()

    # Export plot
    timestamp = ""
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
    path = result_dir_registry[dimension] + f"/{timestamp}training.png"
    plt.savefig(path)
    plt.close()
