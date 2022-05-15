import logging
import os
import sys

import yaml  # type: ignore

from bunruija import options


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("bunruija_cli.gen_yaml")


def infer_vectorizer(model_type):
    vectorizer_dict = {
        "type": None,
        "args": {"tokenizer": {"type": "mecab", "args": {"lemmatize": True}}},
    }

    if model_type in ["lstm", "transformer", "prado", "qrnn"]:
        vectorizer_dict["type"] = "sequence"
    else:
        vectorizer_dict["type"] = "tfidf"

    if model_type == "transformer":
        vectorizer_dict["args"]["tokenizer"]["type"] = "auto"
        vectorizer_dict["args"]["tokenizer"]["args"] = {
            "from_pretrained": "cl-tohoku/bert-base-japanese"
        }

    return vectorizer_dict


def infer_model_args(model_type):
    args = {}
    if model_type == "transformer":
        args["from_pretrained"] = "cl-tohoku/bert-base-japanese"
    return args


def main(args):
    parser = options.get_default_gen_yaml_parser()
    args = parser.parse_args(args)

    setting = {
        "data": {
            "train": None,
            "dev": None,
            "test": None,
        },
        "pipeline": [
            infer_vectorizer(args.model),
            {
                "type": args.model,
                "args": infer_model_args(args.model),
            },
        ],
    }

    if os.path.exists(args.yaml):
        raise FileExistsError(args.yaml)

    with open(args.yaml, "w") as f:
        yaml.dump(setting, f)
    print(setting)


def cli_main():
    main(sys.argv)
