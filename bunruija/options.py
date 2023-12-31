import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="Setting file")
    return parser


def get_default_preprocessing_parser():
    parser = get_parser()
    return parser


def get_default_train_parser():
    parser = get_parser()
    return parser


def get_default_prediction_parser():
    parser = get_parser()
    return parser


def get_default_evaluation_parser():
    parser = get_parser()
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--no-evaluate-time",
        action="store_true",
        help="Disable evaluation of prediction time",
    )
    return parser


def get_default_gen_yaml_parser():
    parser = get_parser()
    parser.add_argument(
        "--model",
        default="sklearn.svm.SVC",
    )
    return parser
