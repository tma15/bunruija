import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', help='Setting file')
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
    return parser
