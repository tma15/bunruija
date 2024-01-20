import logging
import sys

from . import Evaluator, options

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("bunruija.evaluate")


def main(args):
    parser = options.get_default_evaluation_parser()
    args = parser.parse_args(args)

    evaluator = Evaluator(args)
    evaluator.evaluate()


def cli_main():
    main(sys.argv[1:])
