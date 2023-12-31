BUNRUIJA_TOKENIZER_REGISTRY: dict = {}


def register_tokenizer(tokenizer_name: str):
    def f(cls):
        BUNRUIJA_TOKENIZER_REGISTRY[tokenizer_name] = cls
        return cls

    return f
