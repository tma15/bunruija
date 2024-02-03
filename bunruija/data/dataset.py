from pathlib import Path

from datasets import Dataset, load_dataset


def load_data(
    data_path: str | Path,
    label_column: str = "label",
    text_column: str = "text",
) -> tuple[list[str], list[str]]:
    if isinstance(data_path, str):
        data_path = Path(data_path)

    labels: list[str] = []
    texts: list[str] = []

    if data_path.suffix in [".csv", ".json", ".jsonl"]:
        suffix: str = data_path.suffix[1:]

        # Because datasets does not support jsonl suffix, convert it to json
        if suffix == "jsonl":
            suffix = "json"

        # When data_files is only a single data_path, data split is "train"
        dataset: Dataset = load_dataset(suffix, data_files=str(data_path))["train"]

        for sample in dataset:
            labels.append(sample[label_column])
            texts.append(sample[text_column])
        return labels, texts

    else:
        raise ValueError(data_path.suffix)
