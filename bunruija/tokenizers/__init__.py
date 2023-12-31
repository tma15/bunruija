from transformers import AutoTokenizer  # type: ignore

from .mecab_tokenizer import MeCabTokenizer
from .space_tokenizer import SpaceTokenizer
from .tokenizer_registry import register_tokenizer

register_tokenizer("auto")(AutoTokenizer)
DEFAULT_TOKENIZER_NAME = "mecab"

__all__ = ["MeCabTokenizer", "SpaceTokenizer", "register_tokenizer"]
