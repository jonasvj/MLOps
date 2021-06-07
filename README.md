MLOps
==============================

Repo for special course in Machine Learning Operations at DTU.

**Prepare data**
```bash
usage: python make_dataset.py <root>

Script for downloading MNIST data and creating train/test splits.

positional arguments:
  root        Directory to place MNIST data folder in.
```

**Training**
```bash
usage: train_model.py [-h] [--lr LR] [--mb_size MB_SIZE] [--epochs EPOCHS]
                      [--dropout DROPOUT] [--model_path MODEL_PATH]
                      [--data_path DATA_PATH] [--fig_path FIG_PATH]

Script for training an image classifier.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR
  --mb_size MB_SIZE
  --epochs EPOCHS
  --dropout DROPOUT
  --model_path MODEL_PATH
  --data_path DATA_PATH
  --fig_path FIG_PATH
```

**Evaluation**
```bash
usage: evaluate_model.py [-h] [--model_path MODEL_PATH]
                         [--data_path DATA_PATH] [--mb_size MB_SIZE]

Script for evaluating pre-trained image classifier.

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --data_path DATA_PATH
  --mb_size MB_SIZE
```
**Prediction**
```bash
usage: predict_model.py [-h] [--model_path MODEL_PATH] [--mb_size MB_SIZE]
                        data_path

Script for predicting with pre-trained image classifier.

positional arguments:
  data_path

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --mb_size MB_SIZE
```
**Visualization of embeddings**
```bash
usage: visualize.py [-h] [--model_path MODEL_PATH] [--data_path DATA_PATH]
                    [--fig_path FIG_PATH] [--mb_size MB_SIZE]

Script for visualizing embeddings created by image classifier

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --data_path DATA_PATH
  --fig_path FIG_PATH
  --mb_size MB_SIZE
```