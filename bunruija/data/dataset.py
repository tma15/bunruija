from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_data(
    data_path: str | Path,
    label_column: str = "label",
    text_column: str | list[str] = "text",
) -> tuple[list[str], list[str] | list[list[str]]]:
    if isinstance(data_path, str):
        data_path = Path(data_path)

    labels: list[str] = []
    texts: list[str] | list[list[str]]
    texts = []  # type: ignore

    if data_path.suffix in [".csv", ".json", ".jsonl"]:
        suffix: str = data_path.suffix[1:]

        # Because datasets does not support jsonl suffix, convert it to json
        if suffix == "jsonl":
            suffix = "json"

        # When data_files is only a single data_path, data split is "train"
        dataset: DatasetDict | Dataset | IterableDataset | IterableDatasetDict = (
            load_dataset(suffix, data_files=str(data_path), split="train")
        )
        assert isinstance(dataset, Dataset)

        for idx, sample in enumerate(dataset):
            labels.append(sample[label_column])

            if isinstance(text_column, str):
                input_example = sample[text_column]
                texts.append(input_example)
            elif isinstance(text_column, list):
                if len(text_column) != 2:
                    raise ValueError(f"{len(text_column)=}")

                input_example = [sample[text_column[0]], sample[text_column[1]]]
                texts.append(input_example)
        return labels, texts

    else:
        raise ValueError(data_path.suffix)
