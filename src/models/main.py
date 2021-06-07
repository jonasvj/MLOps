import sys
import torch
import argparse
import numpy as np
from torch import nn
from data import mnist
from torch import optim
import matplotlib.pyplot as plt
from model import ImageClassifier

def save_checkpoint(filepath, model):
    checkpoint = {
        'height': model.height,
        'width': model.width,
        'channels': model.channels,
        'classes': model.classes,
        'dropout': model.dropout_rate,
        'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = ImageClassifier(
        height=checkpoint['height'],
        width=checkpoint['width'],
        channels=checkpoint['channels'],
        classes=checkpoint['classes'],
        dropout=checkpoint['dropout'])

    model.load_state_dict(checkpoint['state_dict'])
    
    return model

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Script for either training or evaluating',
            usage='python main.py <command>'
        )
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--mb_size', default=64, type=int)
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--dropout', default=0.25, type=float)
        parser.add_argument('--model_path', default='model.pth', type=str)

        args = parser.parse_args(sys.argv[2:])
        
        # Load data
        train_set, _ = mnist()
    
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.mb_size, shuffle=True)

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
            dropout=args.dropout)

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
            print(f'Epoch: {epoch}, Loss: {loss}')

        # Save model
        save_checkpoint(args.model_path, model)

        # Plot loss vs epoch
        fig, ax = plt.subplots()
        ax.plot(train_losses, label='Train accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()


    def evaluate(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--model_path', default='model.pth', type=str)
        parser.add_argument('--mb_size', default=64, type=int)
        args = parser.parse_args(sys.argv[2:])
        
        model = load_checkpoint(args.model_path)
        
        _, test_set = mnist()
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.mb_size, shuffle=True)

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

        print(f'Accuracy of classifier: {accuracy*100}%')


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    