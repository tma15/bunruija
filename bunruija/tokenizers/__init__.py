from .tokenizer import BaseTokenizer
from .mecab_tokenizer import MeCabTokenizer

from bunruija.filters import PosFilter


BUNRUIJA_TOKENIZER_REGISTRY = {
    'mecab': MeCabTokenizer,
}


def build_tokenizer(config):
    tokenizer_type = config.get('preprocess', {}).get('tokenizer', {}).get('type', 'mecab')
    tokenizer_args = config.get('preprocess', {}).get('tokenizer', {}).get('args', {})

    tokenizer = BUNRUIJA_TOKENIZER_REGISTRY[tokenizer_type](**tokenizer_args)
    return tokenizer
