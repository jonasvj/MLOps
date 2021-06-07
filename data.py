import torch
from torchvision import datasets, transforms

def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=True,
        transform=transform)

    test_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=False,
        transform=transform)

    return train_data, test_data
