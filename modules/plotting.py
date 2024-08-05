import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(history):
    # Plot the learning curves.
    pd.DataFrame(history.history).plot()
    plt.title("Model Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.show()

def plot_lr_vs_loss(history):
    # Plot learning rate vs. loss.
    lrs = 1e-5 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs. Loss")
    plt.show()
