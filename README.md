# Bunruija
Bunruija is a text classification toolkit.
Bunruija aims at enabling pre-processing, training and evaluation of text classification models with **minimum coding effort**.
Bunruija is mainly focusing on Japanese though it is also applicable to other languages.

See `example` for understanding how bunruija is easy to use.

## Features
- **Minimum requirements of coding**: bunruija enables users to train and evaluate their models through command lines. Because all experimental settings are stored in a yaml file, users do not have to write codes.
- **Easy to compare neural-based model with non-neural-based model**: because bunruija supports models based on scikit-learn and PyTorch in the same framework, users can easily compare classification accuracies and prediction times of neural- and non-neural-based models.
- **Easy to reproduce the training of a model**: because all hyperparameters of a model are stored in a yaml file, it is easy to reproduce the model.

## Install
```
poetry install
```
