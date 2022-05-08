from ..registry import BUNRUIJA_REGISTRY
from .mecab_tokenizer import MeCabTokenizer
from .space_tokenizer import SpaceTokenizer

from transformers import AutoTokenizer  # type: ignore


BUNRUIJA_REGISTRY["mecab"] = MeCabTokenizer
BUNRUIJA_REGISTRY["space"] = SpaceTokenizer
BUNRUIJA_REGISTRY["auto"] = AutoTokenizer
DEFAULT_TOKENIZER = MeCabTokenizer
