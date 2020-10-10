from bunruija import options
from bunruija import Predictor


def main():
    parser = options.get_default_prediction_parser()
    args = parser.parse_args()

    predictor = Predictor(args.yaml)
    label = predictor('スマホを買った')
    print(label)
