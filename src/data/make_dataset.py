import argparse
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from torchvision import datasets


def directory(path):
    """Checks if input is a valid directory."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("{} is not a directory".format(path))
    return path


def parser():
    """Parses command line"""
    parser = argparse.ArgumentParser(
        description="Script for downloading MNIST data and creating "
        "train/test splits.",
        usage="python make_dataset.py <root>",
    )
    parser.add_argument("root", help="Directory to place MNIST data folder in.")
    args = parser.parse_args()

    return args


def main():
    """Places raw and procssed MNIST data in specified directory."""
    args = parser()
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    datasets.MNIST(root=args.root, download=True, train=True)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
