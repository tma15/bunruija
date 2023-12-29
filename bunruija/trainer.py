import pickle
from pathlib import Path
from typing import List

import numpy as np
import sklearn  # type: ignore

from . import BunruijaConfig, PipelineBuilder, Saver


class Trainer:
    """Trains a text classification model."""

    def __init__(self, config_file: str):
        self.config = BunruijaConfig.from_yaml(config_file)

        data_path: Path = self.config.bin_dir / "data.bunruija"
        with open(data_path, "rb") as f:
            self.data: dict = pickle.load(f)

        self.model = PipelineBuilder(self.config).build()
        self.saver = Saver(self.config)

    def train(self):
        y_train: np.ndarray = self.data["label_train"]
        X_train: List[str] = self.data["data_train"]
        self.model.fit(X_train, y_train)

        self.saver(self.model)

        if "label_dev" in self.data:
            y_dev = self.data["label_dev"]
            X_dev = self.data["data_dev"]
            y_pred = self.model.predict(X_dev)

            fscore = sklearn.metrics.f1_score(y_dev, y_pred, average="micro")
            print(f"F-score on dev: {fscore}")
            report = sklearn.metrics.classification_report(y_pred, y_dev)
            print(report)
