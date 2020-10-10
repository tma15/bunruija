import logging

import bunruija.classifiers.classifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


BUNRUIJA_CLASSIFIER_REGISTRY = {
    'svm': SVC,
    'rf': RandomForestClassifier,
}

logger = logging.getLogger(__name__)


def build_model(config):
    model_type = config.get('classifier', {}).get('type', 'svm')
    model_args = config.get('classifier', {}).get('args', {})
    logger.info(f'model type: {model_type}')
    logger.info(f'model args: {model_args}')

    model = BUNRUIJA_CLASSIFIER_REGISTRY[model_type](**model_args)
    return model
