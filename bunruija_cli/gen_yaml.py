import importlib
import logging
import os
import sys

import ruamel.yaml  # type: ignore

from bunruija import options
from bunruija.classifiers import TransformerClassifier
from bunruija.classifiers.classifier import NeuralBaseClassifier

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("bunruija_cli.gen_yaml")


def infer_vectorizer(model_cls):
    vectorizer_dict = {
        "type": None,
        "args": {
            "tokenizer": {
                "type": "bunruija.tokenizers.mecab_tokenizer.MeCabTokenizer",
                "args": {"lemmatize": True},
            }
        },
    }

    # if model_type in [
    #     "bunruija.classifiers.lstm.LSTMClassifier",
    #     "bunruija.classifiers.transformer.TransformerClassifier",
    #     "bunruija.classifiers.prado.PRADO",
    #     "bunruija.classifiers.qrnn.QRNN",
    # ]:
    if issubclass(model_cls, NeuralBaseClassifier):
        vectorizer_dict[
            "type"
        ] = "bunruija.feature_extraction.sequence.SequenceVectorizer"
    else:
        vectorizer_dict["type"] = "sklearn.feature_extraction.text.TfidfVectorizer"
        vectorizer_dict["args"]["ngram_range"] = (1, 3)

    # if model_type == "bunruija.classifiers.transformer.TransformerClassifier":
    if model_cls == TransformerClassifier:
        vectorizer_dict["args"]["tokenizer"]["type"] = "transformers.AutoTokenizer"
        vectorizer_dict["args"]["tokenizer"]["args"] = {
            "pretrained_model_name_or_path": "cl-tohoku/bert-base-japanese"
        }

    return vectorizer_dict


def infer_model_args(model_cls):
    args = {}
    if model_cls == TransformerClassifier:
        args["pretrained_model_name_or_path"] = "cl-tohoku/bert-base-japanese"
    return args


def validate_model(model_name):
    module_elems: list[str] = model_name.split(".")
    module_name: str = ".".join(module_elems[:-1])
    cls_name: str = module_elems[-1]
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise ValueError(model_name)
    return cls


def main(args):
    parser = options.get_default_gen_yaml_parser()
    args = parser.parse_args(args)

    model_cls = validate_model(args.model)

    setting = {
        "data": {
            "train": "train.csv",
            "dev": "dev.csv",
            "test": "test.csv",
        },
        "pipeline": [
            infer_vectorizer(model_cls),
            {
                "type": args.model,
                "args": infer_model_args(model_cls),
            },
        ],
    }

    if os.path.exists(args.yaml):
        raise FileExistsError(args.yaml)

    yaml = ruamel.yaml.YAML()
    with open(args.yaml, "w") as f:
        yaml.dump(setting, f)
    yaml.dump(setting, sys.stdout)


def cli_main():
    main(sys.argv[1:])


if __name__ == "__main__":
    cli_main()
