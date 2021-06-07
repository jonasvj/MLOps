import argparse

import torch
from src.models.model import ImageClassifier
from torchvision import datasets, transforms


def parser():
    parser = argparse.ArgumentParser(
        description="Script for evaluating pre-trained image classifier."
    )
    parser.add_argument("--model_path", default="models/model.pth", type=str)
    parser.add_argument("--data_path", default="data/", type=str)
    parser.add_argument("--mb_size", default=64, type=int)
    args = parser.parse_args()

    return args


def load_checkpoint(filepath):
    """Loads model."""
    checkpoint = torch.load(filepath)
    model = ImageClassifier(**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["state_dict"])

    return model


def evaluate(args, model):
    """Evaluates model."""
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
        correct_preds, n_samples = 0, 0

        for images, labels in test_loader:
            ps = torch.exp(model.forward(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            correct_preds += torch.sum(equals).item()
            n_samples += images.shape[0]

        accuracy = correct_preds / n_samples

    print(f"Accuracy of classifier: {accuracy*100}%")


def main():
    args = parser()
    model = load_checkpoint(args.model_path)
    evaluate(args, model)


if __name__ == "__main__":
    main()
