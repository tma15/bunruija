import logging
import sys

from bunruija import options
from bunruija.binarizer import Binarizer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('bunruija_cli.preprocess')


def main(args):
    parser = options.get_default_preprocessing_parser()
    args = parser.parse_args(args)

    binarizer = Binarizer(args.yaml)
    binarizer.binarize()


def cli_main():
    main(sys.argv)
