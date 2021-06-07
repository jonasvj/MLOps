import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.models.model import ImageClassifier
from torch import nn, optim
from torchvision import datasets, transforms


def parser():
    """Parses command line."""
    parser = argparse.ArgumentParser(
        description="Script for training an image classifier."
    )
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--mb_size", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--model_path", default="models/model.pth", type=str)
    parser.add_argument("--data_path", default="data/", type=str)
    parser.add_argument(
        "--fig_path", default="reports/figures/train_loss.pdf", type=str
    )

    args = parser.parse_args()

    return args


def save_checkpoint(filepath, model):
    """Saves model."""
    checkpoint = {"kwargs": model.kwargs, "state_dict": model.state_dict()}

    torch.save(checkpoint, filepath)


def train(args):
    """Trains model."""
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = datasets.MNIST(
        args.data_path, download=False, train=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.mb_size, shuffle=True
    )

    if len(train_set.data.shape) == 4:
        N, height, width, channels = train_set.data.shape
    elif len(train_set.data.shape) == 3:
        N, height, width = train_set.data.shape
        channels = 1

    model = ImageClassifier(
        height=height,
        width=width,
        channels=channels,
        classes=len(np.unique(train_set.targets)),
        dropout=args.dropout,
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = list()
    for epoch in range(args.epochs):
        model.train()

        loss = 0
        for images, labels in train_loader:

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            loss += loss.item()

        train_losses.append(loss.item())
        print(f"Epoch: {epoch}, Loss: {loss}")

    return model, train_losses


def plot_loss(losses, fig_path):
    """Plots training losses."""
    fig, ax = plt.subplots()
    ax.plot(losses, label="Train accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(fig_path)


def main():
    args = parser()
    model, train_losses = train(args)
    save_checkpoint(args.model_path, model)
    plot_loss(train_losses, args.fig_path)


if __name__ == "__main__":
    main()
