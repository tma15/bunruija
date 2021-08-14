from logging import getLogger
from pathlib import Path
import pickle

import lightgbm
from sklearn.svm import (
    LinearSVC,
    SVC
)
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..registry import BUNRUIJA_REGISTRY
from ..tokenizers import build_tokenizer
from .classifier import NeuralBaseClassifier
from .lstm import LSTMClassifier
from .prado import PRADO
from .qrnn import QRNN
from .transformer import TransformerClassifier
from . import util


BUNRUIJA_REGISTRY['lgb'] = lightgbm.LGBMClassifier
BUNRUIJA_REGISTRY['linear_svm'] = LinearSVC
BUNRUIJA_REGISTRY['lr'] = LogisticRegression
BUNRUIJA_REGISTRY['lstm'] = LSTMClassifier
# BUNRUIJA_REGISTRY['pipeline'] = Pipeline
BUNRUIJA_REGISTRY['prado'] = PRADO
BUNRUIJA_REGISTRY['qrnn'] = QRNN
BUNRUIJA_REGISTRY['rf'] = RandomForestClassifier
BUNRUIJA_REGISTRY['svm'] = SVC
BUNRUIJA_REGISTRY['stacking'] = StackingClassifier
BUNRUIJA_REGISTRY['transformer'] = TransformerClassifier
BUNRUIJA_REGISTRY['voting'] = VotingClassifier


logger = getLogger(__name__)


class ClassifierBuilder:
    def __init__(self, config):
        self.config = config

    def maybe_need_more_arg(self, estimator_type):
        if estimator_type in ['tfidf', 'sequence']:
            tokenizer = build_tokenizer(self.config)
            return {'tokenizer': tokenizer}
        else:
            return {}

    def build_estimator(self, estimator_data):
        if isinstance(estimator_data, list):
            estimators = [self.build_estimator(s) for s in estimator_data]
            estimator_type = 'pipeline'
            memory = Path(self.config.get('bin_dir', '.')) / 'cache'
#             estimator = BUNRUIJA_REGISTRY[estimator_type](estimators)
            estimator = Pipeline(estimators, memory=str(memory))
        else:
            estimator_type = estimator_data['type']
            estimator_args = estimator_data.get('args', {})
            additional_args = self.maybe_need_more_arg(estimator_type)
            estimator_args.update(additional_args)

            if isinstance(BUNRUIJA_REGISTRY[estimator_type], NeuralBaseClassifier):
                estimator_args['saver'] = self.saver
            estimator = BUNRUIJA_REGISTRY[estimator_type](**estimator_args)
        return estimator_type, estimator
    
    def build(self):
        setting = self.config['classifier']
        self.saver = util.Saver(self.config)

        if isinstance(setting, list):
            model = self.build_estimator(setting)[1]
        elif isinstance(setting, dict):
            model_type = setting['type']
            model_args = setting.get('args', {})
            model_args['saver'] = self.saver

            if model_type in ['stacking', 'voting']:
                estimators = model_args.pop('estimators')
                for i, estimator_data in enumerate(estimators):
                    estimator_type, estimator = self.build_estimator(estimator_data)
                    if not 'estimators' in model_args:
                        model_args['estimators'] = []
                    name = f'{estimator_type}.{i}'
                    model_args['estimators'].append((name, estimator))

            if model_type == 'stacking':
                final_estimator_data = model_args.pop('final_estimator', {})
                final_estimator_type = final_estimator_data.get('type', None)
                if final_estimator_type:
                    final_estimator_args = final_estimator_data.get('args', {})
                    final_estimator = BUNRUIJA_REGISTRY[final_estimator_type](**final_estimator_args)
                else:
                    final_estimator = None

                model_args['final_estimator'] = final_estimator

            logger.info(f'model type: {model_type}')
            logger.info(f'model args: {model_args}')
            model = BUNRUIJA_REGISTRY[model_type](**model_args)
        logger.info(model)

        return model


def build_model(config):
    builder = ClassifierBuilder(config)
    model = builder.build()
    return model
