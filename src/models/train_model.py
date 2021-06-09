import argparse
import numpy as np
import torch
from src.models.model import ImageClassifier
from torch import nn, optim
import wandb
from src.utils import get_data
from src.visualization.visualize import get_embeddings, plot_embeddings
import matplotlib.pyplot as plt

def parser():
    """Parses command line."""
    parser = argparse.ArgumentParser(
        description='Script for training an image classifier.'
    )
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--mb_size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--model_path', default='models/model.pth', type=str)
    parser.add_argument('--data_path', default='data/', type=str)

    args = parser.parse_args()

    return args


def save_checkpoint(filepath, model):
    """Saves model."""
    checkpoint = {'kwargs': model.kwargs, 'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)


def train(args, data_loader, test_steps=None):
    """Trains model."""

    if len(data_loader.dataset.data.shape) == 4:
        N, height, width, channels = data_loader.dataset.data.shape
    elif len(data_loader.dataset.data.shape) == 3:
        N, height, width = data_loader.dataset.data.shape
        channels = 1
    
    model = ImageClassifier(
        height=height,
        width=width,
        channels=channels,
        classes=len(np.unique(data_loader.dataset.targets)),
        dropout=args.dropout,
    )
    wandb.config.update(model.kwargs)

    if test_steps is not None:
        init_weights = model.linear.weight.clone().detach()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    wandb.watch(model, criterion, log='all', log_freq=10)

    batch = 0
    for epoch in range(args.epochs):
        model.train()

        loss = 0
        for images, labels in data_loader:

            optimizer.zero_grad()
            log_ps = model(images)
            batch_loss = criterion(log_ps, labels)
            batch_loss.backward()
            optimizer.step()
            
            batch_loss = batch_loss.item()
            loss += batch_loss

            wandb.log({
                'batch': batch,
                'Batch loss per sample (train)': batch_loss / images.shape[0]},
                step=batch)
            batch += 1

            if batch == test_steps:
                return model, init_weights
        
        print(f'Epoch: {epoch}, Loss: {loss}')
        wandb.log({'epoch': epoch, 'Train loss': loss}, step=batch-1)

    
    torch.onnx.export(model, images, args.model_path + '.onnx')
    wandb.save(args.model_path + '.onnx')

    return model

def evaluate(model, data_loader):
    """Evaluates model."""

    with torch.no_grad():
        model.eval()
        correct_preds, n_samples = 0, 0

        for images, labels in data_loader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            correct_preds += torch.sum(equals).item()
            n_samples += images.shape[0]
        
        accuracy = correct_preds / n_samples
    
    print(f"Accuracy of classifier: {accuracy*100}%")

    return accuracy

def main():
    # Setup
    args = parser()
    wandb.init(project='MLOps test')
    wandb.config.update(args)
    train_loader, test_loader = get_data(args)

    # Train and save
    model = train(args, train_loader)
    save_checkpoint(args.model_path, model)

    # Evaluate
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    wandb.log({'train_accuracy': train_acc, 'test_accuracy': test_acc})

    # Plot embeddings
    embeddings, labels = get_embeddings(args, model, test_loader)
    fig = plot_embeddings(embeddings, labels)
    wandb.log({"Embeddings": wandb.Image(fig)})

    wandb.finish()


if __name__ == '__main__':
    main()
