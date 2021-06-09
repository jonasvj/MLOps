import torch
from torchvision import datasets, transforms

data_path = 'data/'

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.MNIST(
    data_path, download=True, train=True, transform=transform)
test_set = datasets.MNIST(
    data_path, download=True, train=False, transform=transform)


class TestData:

    def test_train_length(self):
        assert len(train_set.data) == 60000


    def test_test_length(self):
        assert len(test_set.data) == 10000


    def test_data_points(self):
        assert train_set.data.shape == torch.Size([60000, 28, 28])
        assert test_set.data.shape == torch.Size([10000, 28, 28])


    def test_labels(self):
        assert all(torch.unique(train_set.targets) == torch.arange(0,10))
        assert all(torch.unique(test_set.targets) == torch.arange(0,10))