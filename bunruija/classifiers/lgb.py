import copy
import logging

import lightgbm as lgb
import numpy as np

from bunruija.classifiers.classifier import BaseClassifier


logger = logging.getLogger(__name__)


class LightGBMClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__()

        self.label_encoder = kwargs['label_encoder']
        num_class = len(self.label_encoder.classes_)

        if 'objective' in kwargs:
            objective = kwargs['objective']
        elif len(self.label_encoder.classes_) == 2:
            objective = 'binary'
        else:
            objective = 'multiclass'

        self.param = {
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'num_leaves': kwargs.get('num_leaves', 31),
            'num_class': num_class,
            'objective': objective,
            'metric': kwargs.get('metric', ''),
        }

    def fit(self, X, y):
        train_data = lgb.Dataset(X, label=y)
        num_round = 10
        self.bst = lgb.train(self.param, train_data, num_round, valid_sets=None)
        return

    def predict(self, X):
        pred_y_prob = self.bst.predict(X, num_iteration=self.bst.best_iteration)
        pred_y = np.argmax(pred_y_prob, axis=1)
        return pred_y

    def get_params(self, deep=False):
        ret = copy.deepcopy(self.param)
        ret['label_encoder'] = self.label_encoder
        return ret
