import torch
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from src import project_dir

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='data/', batch_size=64, num_workers=0, seed=42):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])

        self.dims = (1, 28, 28)
        self.num_classes = 10


    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            data = MNIST(self.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(
                data, [50000, 10000],
                generator=torch.Generator().manual_seed(self.seed))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = MNIST(
                self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)


    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)


    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('MNISTDataModule')
        parser.add_argument('--data_dir',
            default=project_dir + '/data/', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--num_workers', default=0, type=int)

        return parent_parser
    
    @staticmethod
    def from_argparse_args(namespace):
        ns_dict = vars(namespace)
        args = {
            'data_dir': ns_dict.get('data_dir', project_dir + '/data/'),
            'batch_size': ns_dict.get('batch_size', 64),
            'num_workers': ns_dict.get('num_workers', 0)
            }
        return args