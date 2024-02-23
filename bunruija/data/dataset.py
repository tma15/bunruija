from collections import UserDict

import datasets
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def load_data(
    dataset_args: UserDict,
    split: datasets.Split,
    label_column: str = "label",
    text_column: str | list[str] = "text",
) -> tuple[list[str], list[str] | list[list[str]]]:
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict = (
        load_dataset(split=split, **dataset_args)
    )
    assert isinstance(dataset, Dataset)

    labels: list[str] = []
    texts: list[str] | list[list[str]]
    texts = []  # type: ignore

    for idx, sample in enumerate(dataset):
        label: str

        # If feature of label has names attribute, convert label to actual label strings
        if hasattr(dataset.features[label_column], "names"):
            label = dataset.features[label_column].names[sample[label_column]]
        else:
            label = sample[label_column]

        labels.append(label)

        if isinstance(text_column, str):
            input_example = sample[text_column]
            texts.append(input_example)
        elif isinstance(text_column, list):
            if len(text_column) != 2:
                raise ValueError(f"{len(text_column)=}")

            input_example = [sample[text_column[0]], sample[text_column[1]]]
            texts.append(input_example)
    return labels, texts
