from .tokenizer import BaseTokenizer
from .mecab_tokenizer import MeCabTokenizer

from bunruija.filters import PosFilter


def build_tokenizer():
    tokenizer = MeCabTokenizer(
        lemmatize=True,
        filters=[
            PosFilter(exclude_pos=['助詞'])
        ]
    )
    return tokenizer
