import torch
from torchvision import datasets, transforms


def get_data(args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_set = datasets.MNIST(
        args.data_path, download=False, train=True, transform=transform
    )
    test_set = datasets.MNIST(
        args.data_path, download=False, train=False, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.mb_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.mb_size, shuffle=False
    )

    return train_loader, test_loader