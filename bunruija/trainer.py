import datasets
import numpy as np
import sklearn  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from . import BunruijaConfig, PipelineBuilder, Saver
from .data.dataset import load_data


class Trainer:
    """Trains a text classification model."""

    def __init__(self, config_file: str):
        self.config = BunruijaConfig.from_yaml(config_file)

        self.model = PipelineBuilder(self.config).build()
        self.saver = Saver(self.config)

    def train(self):
        labels_train, X_train = load_data(
            self.config.dataset_args,
            split=datasets.Split.TRAIN,
            label_column=self.config.data.label_column,
            text_column=self.config.data.text_column,
        )

        label_encoder = LabelEncoder()
        y_train: np.ndarray = label_encoder.fit_transform(labels_train)

        self.model.fit(X_train, y_train)

        self.saver(self.model, label_encoder)

        labels_dev, X_dev = load_data(
            self.config.dataset_args,
            split=datasets.Split.VALIDATION,
            label_column=self.config.data.label_column,
            text_column=self.config.data.text_column,
        )

        y_dev: np.ndarray = label_encoder.transform(labels_dev)

        y_pred = self.model.predict(X_dev)

        fscore = sklearn.metrics.f1_score(y_dev, y_pred, average="micro")
        print(f"F-score on dev: {fscore}")
        target_names: list[str] = list(label_encoder.classes_)
        report = sklearn.metrics.classification_report(
            y_pred, y_dev, target_names=target_names
        )
        print(report)
