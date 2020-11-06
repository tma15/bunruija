import logging
from pathlib import Path
import pickle

import bunruija.classifiers.classifier

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)

from .lstm import LSTMClassifier


BUNRUIJA_CLASSIFIER_REGISTRY = {
    'svm': SVC,
    'rf': RandomForestClassifier,
    'lstm': LSTMClassifier,
    'voting': VotingClassifier,
}

logger = logging.getLogger(__name__)


def build_model(config):
    model_type = config.get('classifier', {}).get('type', 'svm')
    model_args = config.get('classifier', {}).get('args', {})
    logger.info(f'model type: {model_type}')
    logger.info(f'model args: {model_args}')

    if model_type in ['lstm']:
        with open(Path(config.get('bin_dir', '.')) / 'model.bunruija', 'rb') as f:
            model_data = pickle.load(f)
            model_args['vectorizer'] = model_data['vectorizer']
            model_args['label_encoder'] = model_data['label_encoder']

    if model_type == 'voting':
        estimators = model_args.pop('estimators')
        for estimator_data in estimators:
            estimator_type = estimator_data['type']
            estimator_args = estimator_data.get('args', {})
            estimator = BUNRUIJA_CLASSIFIER_REGISTRY[estimator_type](**estimator_args)
            if not 'estimators' in model_args:
                model_args['estimators'] = []
            model_args['estimators'].append((estimator_type, estimator))
        logger.info(model_args)

    model = BUNRUIJA_CLASSIFIER_REGISTRY[model_type](**model_args)
    return model
