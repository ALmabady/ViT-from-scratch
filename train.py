# train.py
from typing import Tuple
import torch
from tqdm import tqdm
from torch import nn, optim


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    bar = tqdm(loader, desc="Train", leave=False)
    for images, labels in bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()
        running_correct += batch_correct
        running_samples += labels.size(0)
        running_loss += loss.item()

        acc = batch_correct / labels.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")

    avg_loss = running_loss / len(loader)
    avg_acc = running_correct / running_samples
    return avg_loss, avg_acc


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():
        bar = tqdm(loader, desc="Val", leave=False)
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_samples += labels.size(0)
            val_loss += loss.item()

            bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = val_loss / len(loader)
    avg_acc = val_correct / val_samples
    return avg_loss, avg_acc