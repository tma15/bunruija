import logging
from pathlib import Path
import pickle

import bunruija.classifiers.classifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .lstm import LSTMClassifier


BUNRUIJA_CLASSIFIER_REGISTRY = {
    'svm': SVC,
    'rf': RandomForestClassifier,
    'lstm': LSTMClassifier,
}

logger = logging.getLogger(__name__)


def build_model(config):
    model_type = config.get('classifier', {}).get('type', 'svm')
    model_args = config.get('classifier', {}).get('args', {})
    logger.info(f'model type: {model_type}')
    logger.info(f'model args: {model_args}')

    with open(Path(config.get('bin_dir', '.')) / 'model.bunruija', 'rb') as f:
        model_data = pickle.load(f)

    model = BUNRUIJA_CLASSIFIER_REGISTRY[model_type](
        vectorizer=model_data['vectorizer'],
        label_encoder=model_data['label_encoder'],
        **model_args,
    )
    return model
