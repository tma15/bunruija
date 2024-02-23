import json
from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, load_dataset
from loguru import logger  # type: ignore


def write_json(ds: Dataset, name: Path):
    with open(name, "w") as f:
        for sample in ds:
            category: str = ds.features["label"].names[sample["label"]]
            sample_ = {
                "sentence": sample["sentence"],
                "label": category,
            }
            print(json.dumps(sample_), file=f)
        logger.info(f"{name}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", default="example/jglue/marc_ja/data/jsonl", type=Path
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

    write_json(dataset["train"], args.output_dir / "train.jsonl")
    write_json(dataset["validation"], args.output_dir / "validation.jsonl")


if __name__ == "__main__":
    main()
