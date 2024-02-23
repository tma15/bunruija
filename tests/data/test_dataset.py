import csv
import json
import tempfile
from pathlib import Path

import datasets
import pytest

from bunruija.data.dataset import load_data


def _create_csv_data(
    data_file: Path, samples: list[dict], label_column: str, text_column: str
):
    with open(data_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow([label_column, text_column])
        for sample in samples:
            writer.writerow([sample[label_column], sample[text_column]])


def _create_json_data(data_file: Path, samples):
    with open(data_file, "w") as f:
        print(json.dumps(samples), file=f)


def _create_jsonl_data(data_file: Path, samples):
    with open(data_file, "w") as f:
        for sample in samples:
            print(json.dumps(sample), file=f)


def create_dummy_data(samples, data_file: Path, label_column: str, text_column: str):
    if data_file.suffix == ".csv":
        _create_csv_data(data_file, samples, label_column, text_column)
    elif data_file.suffix == ".jsonl":
        _create_jsonl_data(data_file, samples)
    elif data_file.suffix == ".json":
        _create_json_data(data_file, samples)


@pytest.mark.parametrize("suffix", ["csv", "jsonl", "json"])
def test_load_data(suffix):
    samples = [
        {
            "label": "A",
            "text": "text 1",
        },
        {
            "label": "B",
            "text": "text 2",
        },
        {
            "label": "C",
            "text": "text 3",
        },
    ]

    with tempfile.TemporaryDirectory("test_load_data") as data_dir:
        data_dir = Path(data_dir) / suffix
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        for split in ["train", "validation", "test"]:
            data_file = data_dir / (f"{split}." + suffix)
            create_dummy_data(samples, data_file, "label", "text")

        labels, texts = load_data(
            dataset_args={"path": str(data_dir)},
            label_column="label",
            text_column="text",
            split=datasets.Split.TRAIN,
        )

        assert labels == [sample["label"] for sample in samples]
        assert texts == [sample["text"] for sample in samples]
