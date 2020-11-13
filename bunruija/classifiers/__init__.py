import logging
from pathlib import Path
import pickle

import bunruija.classifiers.classifier

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression

from .lgb import LightGBMClassifier
from .lstm import LSTMClassifier


BUNRUIJA_CLASSIFIER_REGISTRY = {
    'svm': SVC,
    'rf': RandomForestClassifier,
    'lgb': LightGBMClassifier,
    'lr': LogisticRegression,
    'lstm': LSTMClassifier,
    'stacking': StackingClassifier,
    'voting': VotingClassifier,
}

logger = logging.getLogger(__name__)


def build_model(config):
    model_type = config.get('classifier', {}).get('type', 'svm')
    model_args = config.get('classifier', {}).get('args', {})

    additional_args = {}
    with open(Path(config.get('bin_dir', '.')) / 'model.bunruija', 'rb') as f:
        model_data = pickle.load(f)
        additional_args['vectorizer'] = model_data['vectorizer']
        additional_args['label_encoder'] = model_data['label_encoder']

    if model_type in ['stacking', 'voting']:
        estimators = model_args.pop('estimators')
        for estimator_data in estimators:
            estimator_type = estimator_data['type']
            estimator_args = estimator_data.get('args', {})

            if estimator_type in ['lgb', 'lstm']:
                estimator_args['vectorizer'] = additional_args['vectorizer']
                estimator_args['label_encoder'] = additional_args['label_encoder']
            estimator = BUNRUIJA_CLASSIFIER_REGISTRY[estimator_type](**estimator_args)
            if not 'estimators' in model_args:
                model_args['estimators'] = []
            model_args['estimators'].append((estimator_type, estimator))

    if model_type == 'stacking':
        final_estimator_data = model_args.pop('final_estimator', {})
        final_estimator_type = final_estimator_data.get('type', None)
        if final_estimator_type:
            final_estimator_args = final_estimator_data.get('args', {})
            final_estimator = BUNRUIJA_CLASSIFIER_REGISTRY[final_estimator_type](**final_estimator_args)
        else:
            final_estimator = None

        model_args['final_estimator'] = final_estimator

    if model_type in ['lgb', 'lstm']:
        model_args['vectorizer'] = additional_args['vectorizer']
        model_args['label_encoder'] = additional_args['label_encoder']

    logger.info(f'model type: {model_type}')
    logger.info(f'model args: {model_args}')
    model = BUNRUIJA_CLASSIFIER_REGISTRY[model_type](**model_args)
    return model
