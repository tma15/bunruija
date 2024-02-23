import csv
import random
import tempfile
import unittest
from pathlib import Path

import ruamel.yaml  # type: ignore
import torch

from bunruija import evaluate, gen_yaml, train


def create_dummy_data(data_dir, num_samples=5, num_labels=3, max_len=100):
    K = 3
    labels_data = torch.rand(num_samples * K)
    labels_data = 97 + torch.floor(26 * labels_data).int()
    offset = 0
    labels = []
    for i in range(num_labels):
        label_len = random.randint(1, K)
        label_str = "".join(map(chr, labels_data[offset : offset + label_len]))
        labels.append(label_str)
        offset += label_len

    def _create_dummy_data(filename):
        with open(Path(data_dir) / filename, "w") as f:
            offset = 0

            data = torch.rand(num_samples * max_len)
            data = 97 + torch.floor(26 * data).int()

            writer = csv.writer(f)
            writer.writerow(["label", "text"])
            for i in range(num_labels):
                label = labels[i]
                sample_len = random.randint(30, max_len)
                sample_str = "".join(map(chr, data[offset : offset + sample_len]))
                offset += sample_len
                writer.writerow([label, sample_str])

            for i in range(num_samples - num_labels):
                label = random.choice(labels)
                sample_len = random.randint(30, max_len)
                sample_str = "".join(map(chr, data[offset : offset + sample_len]))
                offset += sample_len
                writer.writerow([label, sample_str])

    _create_dummy_data("train.csv")
    _create_dummy_data("validation.csv")
    _create_dummy_data("test.csv")


class TestBinary(unittest.TestCase):
    def rewrite_data_path(self, data_dir, yaml_file):
        yaml = ruamel.yaml.YAML()

        with open(yaml_file, "r") as f:
            setting = yaml.load(f)
            setting["data"]["args"]["path"] = str(data_dir)
            setting["output_dir"] = str(Path(data_dir) / "output_dir")

        with open(yaml_file, "w") as f:
            yaml.dump(setting, f)

    def execute(self, model):
        with tempfile.TemporaryDirectory(f"test_{model}") as data_dir:
            data_dir = Path(data_dir) / "csv"
            if not data_dir.exists():
                data_dir.mkdir(parents=True)

            create_dummy_data(data_dir)
            yaml_file = str(Path(data_dir) / "test-binary.yaml")

            gen_yaml.main(
                [
                    "--model",
                    model,
                    "-y",
                    yaml_file,
                ]
            )

            self.rewrite_data_path(data_dir, yaml_file)

            train.main(["-y", yaml_file])
            evaluate.main(["-y", yaml_file])

    def test_svm(self):
        self.execute("sklearn.svm.SVC")

    def test_lstm(self):
        self.execute("bunruija.classifiers.lstm.LSTMClassifier")

    def test_transformer(self):
        self.execute("bunruija.classifiers.transformer.TransformerClassifier")
