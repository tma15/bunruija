import csv
import json
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from loguru import logger  # type: ignore


def write_csv(samples: list[dict], name: Path):
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        for sample in samples:
            writer.writerow([f"{sample['category']}", sample["title"]])
        logger.info(f"{name}")


def write_json(samples: list[dict], name: Path):
    with open(name, "w") as f:
        for sample in samples:
            sample_ = {
                "text": sample["title"],
                "label": sample["category"],
            }
            print(json.dumps(sample_), file=f)
        logger.info(f"{name}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", default="example/livedoor_corpus", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(
        "shunk031/livedoor-news-corpus",
        random_state=0,
        shuffle=True,
    )
    write_csv(dataset["train"], args.output_dir / "train.csv")
    write_csv(dataset["validation"], args.output_dir / "dev.csv")
    write_csv(dataset["test"], args.output_dir / "test.csv")

    write_json(dataset["train"], args.output_dir / "train.jsonl")
    write_json(dataset["validation"], args.output_dir / "dev.jsonl")
    write_json(dataset["test"], args.output_dir / "test.jsonl")


if __name__ == "__main__":
    main()
