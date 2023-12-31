import pickle
from argparse import Namespace

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .dataclass import BunruijaConfig
from .predictor import Predictor


class Evaluator:
    """Evaluates a trained model"""

    def __init__(self, args: Namespace):
        self.config = BunruijaConfig.from_yaml(args.yaml)

        self.verbose = args.verbose

        self.predictor = Predictor(args.yaml)

        data_path = self.config.bin_dir / "data.bunruija"
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def evaluate(self):
        X_test: list[str] = self.data["data_test"]
        y_test: np.ndarray = self.data["label_test"]
        y_pred: np.ndarray = self.predictor(X_test)

        labels = list(self.predictor.label_encoder.classes_)
        if self.verbose:
            conf_mat = confusion_matrix(y_test, y_pred)
            for i in range(len(labels)):
                print("True", "Pred", "Num samples", sep="\t")
                for j in range(len(labels)):
                    print(labels[i], labels[j], conf_mat[i, j], sep="\t")
                print()

        report: str = classification_report(
            y_test,
            y_pred,
            target_names=labels,
        )
        print(report)
