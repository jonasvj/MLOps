import torch
import wandb
from argparse import Namespace
from src.models.train_model import train, evaluate
from src.utils import get_data

wandb.init(mode='disabled')

args_dict = {
        'epochs': 2,
        'dropout': 0.25,
        'lr': 3e-4,
        'mb_size': 64,
        'data_path': 'data/'
    }
args = Namespace(**args_dict)
train_loader, test_loader = get_data(args)

class TestTrain:

    def test_weights(self):
        model, init_weights = train(args, train_loader, test_steps=2)
        assert not torch.all(torch.eq(init_weights, model.linear.weight))
    
    def test_evaluate(self):
        model, init_weights = train(args, train_loader, test_steps=1)
        accuracy = evaluate(model, test_loader)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy and accuracy <= 1