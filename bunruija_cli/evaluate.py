import logging
import sys

from bunruija import options
from bunruija import Evaluator


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('bunruija_cli.evaluate')


def main():
    parser = options.get_default_evaluation_parser()
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.evaluate()
