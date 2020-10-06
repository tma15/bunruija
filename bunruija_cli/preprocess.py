import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from bunruija import options
from bunruija.binarizer import Binarizer


def main():
    parser = options.get_default_preprocessing_parser()
    args = parser.parse_args()
    print(args)

    config_file = None
    input_file = None
    binarizer = Binarizer(config_file)
    binarizer.binarize(input_file)
