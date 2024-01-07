import csv
from pathlib import Path


def load_data(data_path: str | Path) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    texts: list[str] = []
    with open(data_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if len(row[0]) == 0 or len(row[1]) == 0:
                continue
            labels.append(row[0])
            texts.append(row[1])
    return labels, texts
