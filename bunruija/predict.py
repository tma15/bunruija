import sys

from . import Predictor, options


def main(args):
    parser = options.get_default_prediction_parser()
    args = parser.parse_args(args)

    predictor = Predictor.from_pretrained(args.model)
    while True:
        text = input("Input:")
        label: list[str] = predictor([text], return_label_type="str")
        print(label[0])


def cli_main():
    main(sys.argv[1:])


if __name__ == "__main__":
    cli_main()
