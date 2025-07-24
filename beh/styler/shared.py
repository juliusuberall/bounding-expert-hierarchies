import jax
import matplotlib.pyplot as plt

from beh.registry import *

# Plot the gradient of a neural network training
def checkpoint_moe_export_plot_gradient(gradient, dimension, epoch):
  
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