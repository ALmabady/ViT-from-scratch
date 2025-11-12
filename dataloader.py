# dataloader.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def get_dataloaders(
    data_root: str = "./data",
    img_size: int = 32,
    batch_size: int = 128,
    download: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Returns: train_loader, val_loader, num_classes, in_channels
    """
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = MNIST(root=data_root, train=True, download=download, transform=transform)
    val_ds = MNIST(root=data_root, train=False, download=download, transform=transform)

    # infer dataset properties
    try:
        in_channels = train_ds[0][0].shape[0]
    except Exception:
        in_channels = 1

    # number of classes (robust)
    if hasattr(train_ds, "targets"):
        num_classes = int(torch.unique(train_ds.targets).numel())
    else:
        # fallback
        num_classes = 10

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, num_classes, in_channels
