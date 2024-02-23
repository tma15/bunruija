import json
from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, load_dataset
from loguru import logger  # type: ignore


def write_json(ds: Dataset, name: Path):
    with open(name, "w") as f:
        for sample in ds:
            category: str = ds.features["category"].names[sample["category"]]
            sample_ = {
                "title": sample["title"],
                "category": category,
            }
            print(json.dumps(sample_), file=f)
        logger.info(f"{name}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", default="example/livedoor_corpus/data/jsonl", type=Path
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "shunk031/livedoor-news-corpus",
        random_state=0,
        shuffle=True,
    )

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    write_json(dataset["train"], args.output_dir / "train.jsonl")
    write_json(dataset["validation"], args.output_dir / "validation.jsonl")
    write_json(dataset["test"], args.output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
