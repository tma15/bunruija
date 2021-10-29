import sys

from bunruija import options
from bunruija import Predictor


def main(args):
    parser = options.get_default_prediction_parser()
    args = parser.parse_args(args)

    predictor = Predictor(args.yaml)
    while True:
        text = input()
        label = predictor(text)
        print(label)


def cli_main():
    main(sys.argv[1:])
