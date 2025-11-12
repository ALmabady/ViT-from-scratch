# utils.py
import random
import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def set_seed(seed: int = 23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_predictions(images, preds, labels, n: int = 8, save_path: str = "predictions.png"):
    """
    images: tensor (B, C, H, W) in CPU
    preds, labels: cpu tensors
    """
    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    n = min(n, images.shape[0])
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, (n + 1) // 2, i + 1)
        img = images[i].squeeze()
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            # CHW -> HWC
            plt.imshow(img.permute(1, 2, 0))
        plt.title(f"P:{int(preds[i])} | L:{int(labels[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
