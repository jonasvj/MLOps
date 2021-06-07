import argparse
import pickle

import torch
from src.models.model import ImageClassifier


def parser():
    """Parses command line."""
    parser = argparse.ArgumentParser(
        description="Script for predicting with pre-trained image classifier."
    )
    parser.add_argument("data_path", type=str)
    parser.add_argument("--model_path", default="models/model.pth", type=str)
    parser.add_argument("--mb_size", default=64, type=int)
    args = parser.parse_args()

    return args


def load_checkpoint(filepath):
    """Loads model."""
    checkpoint = torch.load(filepath)
    model = ImageClassifier(**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["state_dict"])

    return model


def load_data(filepath):
    """Loads data. Assumes data is an np array saved as pickle."""
    with open(filepath, "rb") as f:
        x = pickle.load(f)

    x = torch.from_numpy(x)
    x = torch.utils.data.TensorDataset(x)

    return x


def predict(args, model, data):
    """Predicts with model."""

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.mb_size, shuffle=False
    )

    probs = torch.zeros((len(data), model.classes))

    with torch.no_grad():
        model.eval()
        for i, images in enumerate(data_loader):
            ps = torch.exp(model.forward(images))
            probs[i*args.mb_size:i*args.mb_size+images.shape[0],:] = ps

    return probs


def main():
    args = parser()
    model = load_checkpoint(args.model_path)
    data = load_data(args.data_path)
    probs = predict(args, model, data)
    print(probs)


if __name__ == "__main__":
    main()
