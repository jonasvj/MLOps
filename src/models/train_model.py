import argparse
from src.models.model import ConvNet, ImageClassifier
from src.data.mnist import MNISTDataModule
from pytorch_lightning.loggers import WandbLogger
from src import project_dir
import pytorch_lightning as pl

def parser(lightning_class, data_class, model_class):
    """Parses command line."""
    parser = argparse.ArgumentParser()

    # Progam level args
    parser.add_argument('--project', default='MNIST', type=str)
    parser.add_argument(
        '--model_path', default=project_dir + '/models/model.pth', type=str)
  
    # Training level args
    parser = pl.Trainer.add_argparse_args(parser)

    # Lightning level args
    parser = lightning_class.add_model_specific_args(parser)
    
    # Data level args
    parser = data_class.add_model_specific_args(parser)

    # Model level args
    parser = model_class.add_model_specific_args(parser)

    args = parser.parse_args()

    return args

def main():
    # Setup
    args = parser(ImageClassifier, MNISTDataModule, ConvNet)
    wandb_logger = WandbLogger(
        project=args.project, log_model='all', config=args)

    # Data
    dm = MNISTDataModule(**MNISTDataModule.from_argparse_args(args))
    dm.prepare_data()

    # Model
    model = ConvNet(**ConvNet.from_argparse_args(args))
    classifier = ImageClassifier(
        model, **ImageClassifier.from_argparse_args(args))
    wandb_logger.watch(classifier)

    # Trainer
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)

    # Train
    dm.setup(stage='fit')
    trainer.fit(model=classifier, datamodule=dm)
    trainer.save_checkpoint(args.model_path)

    # Test
    dm.setup(stage='test')
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    print('starting')
    main()
