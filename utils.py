import os
import torch
import shutil
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def compute_conv_transposed_output_size(
        input_size: int,
        stride: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0) -> int:
    output_size = ((input_size-1)*stride -
                   2*padding + dilation*(kernel_size-1) + output_padding + 1)
    return output_size


def save_grid(images: np.array, title: str, file_name: str) -> None:

    images = torch.tensor(images,
                          dtype=torch.float32)
    grid = make_grid(images,
                     nrow=4,
                     padding=2,
                     normalize=True,
                     value_range=(-1, 1))
    plt.figure(figsize=(4,4))
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(file_name)

def plot_learning_curves(d_losses: List[float],
                         g_losses: List[float],
                         title: str,
                         file_name: str) -> None:
    plt.figure()
    plt.title(title)
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.legend()
    plt.grid()
    plt.savefig(file_name)
