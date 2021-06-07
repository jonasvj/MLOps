import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from src.models.model import ImageClassifier


def parser():
    """Parses command line."""
    parser = argparse.ArgumentParser(
        description="Script for visualizing embeddings created by image " "classifier"
    )
    parser.add_argument("--model_path", default="models/model.pth", type=str)
    parser.add_argument("--data_path", default="data/", type=str)
    parser.add_argument("--fig_path", default="reports/figures/embeddings.pdf")
    parser.add_argument("--mb_size", default=64, type=int)
    args = parser.parse_args()

    return args


def load_checkpoint(filepath):
    """Loads model."""
    checkpoint = torch.load(filepath)
    model = ImageClassifier(**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["state_dict"])

    return model


def get_embeddings(args, model):
    """Gets embeddings produced by model."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    test_set = datasets.MNIST(
        args.data_path, download=False, train=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.mb_size, shuffle=False
    )

    with torch.no_grad():
        model.eval()
        embeddings = torch.zeros((len(test_set), model.linear.in_features))
        all_labels = torch.zeros(len(test_set))
        for i, (images, labels) in enumerate(test_loader):
            model.forward(images)
            embeddings[
                i * args.mb_size : i * args.mb_size + images.shape[0], :
            ] = model.embeddings
            all_labels[i * args.mb_size : i * args.mb_size + images.shape[0]] = labels

    return embeddings.numpy(), all_labels.numpy()


def plot_embeddings(embeddings, labels, fig_path):
    """Plots embeddings."""
    embs_proj = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        embs_proj[:, 0],
        embs_proj[:, 1],
        c=labels,
        cmap=plt.get_cmap("tab10"),
        alpha=0.5,
        s=2,
    )
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_title("Embeddings")

    markers = scatter.legend_elements()[0]
    plt.legend(
        markers,
        np.unique(labels),
        loc=0,
        borderaxespad=0.1,
        title="Digit",
        framealpha=0.6,
    )

    plt.savefig(fig_path)


def main():
    args = parser()
    model = load_checkpoint(args.model_path)
    embeddings, labels = get_embeddings(args, model)
    plot_embeddings(embeddings, labels, args.fig_path)


if __name__ == "__main__":
    main()
