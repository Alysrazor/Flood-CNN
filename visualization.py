import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

PLOT_DIR = Path(__file__).resolve().parent / "plots"


def save_plots(
    train_acc: list,
    valid_acc: list,
    train_loss: list,
    valid_loss: list,
) -> None:
    """
    This function saves the training and
    validation loss and accuracy plots.
    """

    Path.mkdir(PLOT_DIR, exist_ok=True)

    plt.figure(figsize = (10, 7))
    plt.plot(
        train_acc,
        color='blue',
        linestyle='dashed',
        label='Training Accuracy',
    )
    plt.plot(
        valid_acc,
        color='green',
        linestyle='dashed',
        label='Validation Accuracy',
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(PLOT_DIR / 'Accuracy.png')

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss,
        color='blue',
        linestyle='dashed',
        label='Training Loss',
    )
    plt.plot(
        valid_loss,
        color='green',
        linestyle='dashed',
        label='Validation Loss',
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(PLOT_DIR / 'Loss.png')