import sys
from logging import INFO, basicConfig, getLogger

from . import options
from .trainer import Trainer

basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=INFO,
    stream=sys.stdout,
)
logger = getLogger("bunruija.train")


def main(args):
    parser = options.get_default_train_parser()
    args = parser.parse_args(args)

    trainer = Trainer(args.yaml)
    trainer.train()


def cli_main():
    main(sys.argv[1:])


if __name__ == "__main__":
    cli_main()
