from ..registry import BUNRUIJA_REGISTRY


def build_tokenizer(config):
    tokenizer_type = config.get("type", "mecab")
    tokenizer_args = config.get("args", {})

    if "from_pretrained" in tokenizer_args:
        tokenizer = BUNRUIJA_REGISTRY[tokenizer_type].from_pretrained(
            tokenizer_args["from_pretrained"]
        )
    else:
        tokenizer = BUNRUIJA_REGISTRY[tokenizer_type](**tokenizer_args)
    return tokenizer
