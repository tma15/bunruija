from ..registry import BUNRUIJA_REGISTRY
from .tokenizer import BaseTokenizer
from .mecab_tokenizer import MeCabTokenizer


BUNRUIJA_REGISTRY['mecab'] = MeCabTokenizer


def build_tokenizer(config):
    tokenizer_type = config.get('tokenizer', {}).get('type', 'mecab')
    tokenizer_args = config.get('tokenizer', {}).get('args', {})

    tokenizer = BUNRUIJA_REGISTRY[tokenizer_type](**tokenizer_args)
    return tokenizer


def build_default_tokenizer():
    tokenizer = MeCabTokenizer()
    return tokenizer
