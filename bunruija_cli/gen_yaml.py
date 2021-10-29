import logging
import os
import sys

import yaml

from bunruija.feature_extraction import get_default_vectorizer_setting_by_model
from bunruija import options


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('bunruija_cli.gen_yaml')


def infer_vectorizer(model_type):
    if model_type in ['lstm']:
        return 'sequence'
    else:
        return 'tfidf'


def main(args):
    parser = options.get_default_gen_yaml_parser()
    args = parser.parse_args(args)

    setting = {
        'preprocess': {
            'data': {
                'train': None,
                'dev': None,
                'test': None,
            },
        },
        'tokenizer': {
            'type': 'mecab',
            'args': {
                'lemmatize': True,
            }
        },
        'classifier': [
            {
                'type': infer_vectorizer(args.model),
            },
            {
                'type': args.model,
            },
        ],
    }

    if os.path.exists(args.yaml):
        raise FileExistsError(args.yaml)

    with open(args.yaml, 'w') as f:
        yaml.dump(setting, f)
    print(setting)


def cli_main():
    main(sys.argv)
