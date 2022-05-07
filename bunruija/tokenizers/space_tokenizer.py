from typing import List

from .tokenizer import BaseTokenizer


class SpaceTokenizer(BaseTokenizer):
    def __init__(self, reduce_redundant_spaces: bool = True, **kwargs):
        self.reduce_redundant_spaces = reduce_redundant_spaces
        super().__init__(name="space")

    def __call__(self, text) -> List[str]:
        if self.reduce_redundant_spaces:
            result = [token for token in text.split(" ") if token != ""]
            return result
        else:
            return text.split(" ")

    def __repr__(self) -> str:
        out = f"{self.__class__.__name__}()"
        return out
