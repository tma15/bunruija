import os
from pathlib import Path
import pickle

import sklearn
import torch
import yaml

import bunruija


class Trainer:
    """Trains a text classification model.
    """
    def __init__(self, config_file: str):
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        with open(Path(self.config.get('bin_dir', '.')) / 'data.bunruija', 'rb') as f:
            self.data = pickle.load(f)

        self.model = bunruija.classifiers.build_model(self.config)
        self.saver = bunruija.classifiers.util.Saver(self.config)

    def train(self):
        y_train = self.data['label_train']
        X_train = self.data['data_train']
        self.model.fit(X_train, y_train)

        self.saver(self.model)

        if 'label_dev' in self.data:
            y_dev = self.data['label_dev']
            X_dev = self.data['data_dev']
            y_pred = self.model.predict(X_dev)

            fscore = sklearn.metrics.f1_score(y_dev, y_pred, average='micro')
            print(f'F-score on dev: {fscore}')
            report = sklearn.metrics.classification_report(y_pred, y_dev)
            print(report)
