import functools
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from bunruija.tokenizers import build_tokenizer

from ..registry import BUNRUIJA_REGISTRY
from .sequence import SequenceVectorizer


BUNRUIJA_REGISTRY['sequence'] = SequenceVectorizer
BUNRUIJA_REGISTRY['tfidf'] = TfidfVectorizer

logger = logging.getLogger(__name__)


# https://stackoverflow.com/questions/9336646/python-decorator-with-multiprocessing-fails
def register_vectorizer(vectorizer_name, vectorizer):
    if vectorizer_name in BUNRUIJA_REGISTRY:
        raise KeyError
    BUNRUIJA_REGISTRY[vectorizer_name] = vectorizer

# def register_vectorizer(vectorizer_name):
#     def f(vectorizer):
#         class new_vectorizer(vectorizer):
#             def fit(self, raw_documents):
#                 return super().fit(raw_documents)

#             def transforme(self, raw_documents):
#                 return super().transform(raw_documents)

#         new_vectorizer.__name__ = vectorizer.__name__
#         BUNRUIJA_VECTORIZER_REGISTRY[vectorizer_name] = new_vectorizer
#     return f

# def register_vectorizer(vectorizer_name):
#     def f(vectorizer):
#         print(vectorizer)
#         BUNRUIJA_VECTORIZER_REGISTRY[vectorizer_name] = vectorizer
#     return f

# class register_vectorizer:
#     def __init__(self, vectorizer_name):
#         self.vectorizer_name = vectorizer_name

#     def __call__(self, vectorizer):
#         class new_vectorizer(vectorizer):
#             def fit(self, raw_documents):
#                 return super().fit(raw_documents)

#             def transforme(self, raw_documents):
#                 return super().transform(raw_documents)

#         new_vectorizer.__name__ = vectorizer.__name__
#         BUNRUIJA_VECTORIZER_REGISTRY[self.vectorizer_name] = new_vectorizer


def build_vectorizer(config, tokenizer=None):
    vectorizer_setting = config.get('preprocess', {}).get('vectorizer', {})

    vectorizers = []
    if isinstance(vectorizer_setting, list):
        for vs in vectorizer_setting:
            vectorizer_type = vs.get('type', 'tfidf')
            vectorizer_args = vs.get('args', {})
            vectorizer_name = vs.get('name', vectorizer_type)
            logger.info(f'vectorizer type: {vectorizer_type}')
            logger.info(f'vectorizer args: {vectorizer_args}')

            if vectorizer_type == 'tfidf':
                vectorizer = BUNRUIJA_REGISTRY[vectorizer_type](
                    tokenizer=build_tokenizer(config),
                    **vectorizer_args)
            else:
                vectorizer = BUNRUIJA_REGISTRY[vectorizer_type](
                    **vectorizer_args)
            vectorizers.append((vectorizer_name, vectorizer))
    else:
        vectorizer_type = vectorizer_setting.get('type', 'tfidf')
        vectorizer_args = vectorizer_setting.get('args', {})
        logger.info(f'vectorizer type: {vectorizer_type}')
        logger.info(f'vectorizer args: {vectorizer_args}')

        if vectorizer_type in ['tfidf', 'sequence']:
            vectorizer = BUNRUIJA_REGISTRY[vectorizer_type](
                tokenizer=build_tokenizer(config),
                **vectorizer_args)
        else:
            vectorizer = BUNRUIJA_REGISTRY[vectorizer_type](
                **vectorizer_args)
        vectorizers = [(vectorizer_type, vectorizer)]

    vectorizer = FeatureUnion(vectorizers)

    return vectorizer


def get_default_vectorizer_setting_by_model(model):
    if model in ['lstm']:
        setting = {
            'type': 'sequence',
            'args': {
                'max_features': 10000,
                'min_df': 1,
            }
        }
    else:
        setting = {
            'type': 'tfidf',
            'args': {
                'max_features': 10000,
                'min_df': 1,
            }
        }
    return setting

