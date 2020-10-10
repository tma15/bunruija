from .tokenizer import BaseTokenizer
from .mecab_tokenizer import MeCabTokenizer

from bunruija.filters import PosFilter


def build_tokenizer(config):
    lemmatize = config.get('preprocess', {}).get('tokenizer', {}).get('lemmatize', False)
    exclude_pos = config.get('preprocess', {}).get('tokenizer', {}).get('exclude_pos', [])

    tokenizer = MeCabTokenizer(
        lemmatize=lemmatize,
        filters=[
            PosFilter(exclude_pos=exclude_pos)
        ]
    )
    return tokenizer
