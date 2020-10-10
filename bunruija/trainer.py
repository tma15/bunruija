import logging
import pickle
import yaml
from pathlib import Path

import sklearn

import bunruija


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        with open(Path(self.config.get('bin_dir', '.')) / 'data.bunruija', 'rb') as f:
            self.data = pickle.load(f)

        self.model = bunruija.classifiers.build_model(self.config)

    def train(self):
        y_train = self.data['label_train']
        X_train = self.data['data_train']
        self.model.fit(X_train, y_train)

        y_dev = self.data['label_dev']
        X_dev = self.data['data_dev']
        y_pred = self.model.predict(X_dev)

        with open(Path(self.config.get('bin_dir', '.')) / 'model.bunruija', 'rb') as f:
            model_data = pickle.load(f)

        with open(Path(self.config.get('bin_dir', '.')) / 'model.bunruija', 'wb') as f:
            model_data['classifier'] = self.model
            pickle.dump(model_data, f)

        fscore = sklearn.metrics.f1_score(y_dev, y_pred, average='micro')
        logger.info(f'F-score on dev: {fscore}')
