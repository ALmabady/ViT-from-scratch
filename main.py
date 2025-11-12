# main.py
import argparse
import os
import torch
from torch import nn, optim

from dataloader import get_dataloaders
from model import VisionTransformer
from train import train_one_epoch, validate
from utils import set_seed, get_device, plot_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Train a small ViT on MNIST")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="./models")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, num_classes, in_channels = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        download=True,
    )

    model = VisionTransformer(
        image_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc*100:.2f}%")

        # save checkpoint
        ckpt_path = f"{args.save_dir}/vit_epoch{epoch+1}.pth"
        torch.save({"epoch": epoch + 1, "model_state": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)

    # show sample predictions
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(val_loader))
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    # move tensors to cpu for plotting
    plot_predictions(images.cpu(), preds.cpu(), labels.cpu(), n=8)
    print("Done, Full code Ran successfully")


if __name__ == "__main__":
    main()
