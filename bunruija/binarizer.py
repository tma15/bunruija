import csv
import os
from pathlib import Path
import pickle
from typing import Tuple
import yaml

from sklearn.preprocessing import LabelEncoder


class Binarizer:
    """Binarizes data"""

    def __init__(self, config_file: str):
        self.config_file = config_file
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        if not os.path.exists(self.config.get("bin_dir", ".")):
            os.makedirs(self.config.get("bin_dir", "."))

    def load_data(self, data_path: str) -> Tuple[str, str]:
        labels = []
        texts = []
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

    def binarize(self):
        labels_train, texts_train = self.load_data(self.config["data"]["train"])

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(labels_train)

        if "dev" in self.config["data"]:
            labels_dev, texts_dev = self.load_data(self.config["data"]["dev"])
            y_dev = label_encoder.transform(labels_dev)

        if "test" in self.config["data"]:
            labels_test, texts_test = self.load_data(self.config["data"]["test"])
            y_test = label_encoder.transform(labels_test)

        with open(Path(self.config.get("bin_dir", ".")) / "data.bunruija", "wb") as f:
            data = {
                "label_train": y_train,
                "data_train": texts_train,
            }

            if "dev" in self.config["data"]:
                data["label_dev"] = y_dev
                data["data_dev"] = texts_dev

            if "test" in self.config["data"]:
                data["label_test"] = y_test
                data["data_test"] = texts_test
            pickle.dump(data, f)

        with open(Path(self.config.get("bin_dir", ".")) / "model.bunruija", "wb") as f:
            pickle.dump(
                {
                    "label_encoder": label_encoder,
                },
                f,
            )
