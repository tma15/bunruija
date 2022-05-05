from bunruija.tokenizers import BaseTokenizer


class SpaceTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__(name="space")

    def __call__(self, text):
        result = text.split(" ")
        return result

    def __repr__(self):
        out = f"{self.__class__.__name__}()"
        return out
