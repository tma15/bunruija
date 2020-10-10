import logging

from sklearn.feature_extraction.text import TfidfVectorizer

from bunruija.tokenizers import build_tokenizer

from .sequence import SequenceVectorizer


BUNRUIJA_VECTORIZER_REGISTRY = {
    'sequence': SequenceVectorizer,
    'tfidf': TfidfVectorizer,
}

logger = logging.getLogger(__name__)


def build_vectorizer(config, tokenizer=None):
    vectorizer_type = config.get('preprocess', {}).get('vectorizer', {}).get('type', 'tfidf')
    vectorizer_args = config.get('preprocess', {}).get('vectorizer', {}).get('args', {})
    logger.info(f'vectorizer type: {vectorizer_type}')
    logger.info(f'vectorizer args: {vectorizer_args}')

    vectorizer = BUNRUIJA_VECTORIZER_REGISTRY[vectorizer_type](
        tokenizer=build_tokenizer(config),
        **vectorizer_args)
    return vectorizer
