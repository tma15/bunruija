from ..registry import BUNRUIJA_REGISTRY
from .tokenizer import BaseTokenizer
from .mecab_tokenizer import MeCabTokenizer
from .space_tokenizer import SpaceTokenizer

from transformers import AutoTokenizer


BUNRUIJA_REGISTRY['mecab'] = MeCabTokenizer
BUNRUIJA_REGISTRY['space'] = SpaceTokenizer
BUNRUIJA_REGISTRY['auto'] = AutoTokenizer


def build_tokenizer(config):
    tokenizer_type = config.get('tokenizer', {}).get('type', 'mecab')
    tokenizer_args = config.get('tokenizer', {}).get('args', {})

    if 'from_pretrained' in tokenizer_args:
        tokenizer = BUNRUIJA_REGISTRY[tokenizer_type].from_pretrained(
            tokenizer_args['from_pretrained'])
    else:
        tokenizer = BUNRUIJA_REGISTRY[tokenizer_type](**tokenizer_args)
    return tokenizer


def build_default_tokenizer():
    tokenizer = MeCabTokenizer()
    return tokenizer
