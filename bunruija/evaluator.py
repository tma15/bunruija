from argparse import Namespace

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .data.dataset import load_data
from .dataclass import BunruijaConfig
from .predictor import Predictor


class Evaluator:
    """Evaluates a trained model"""

    def __init__(self, args: Namespace):
        self.config = BunruijaConfig.from_yaml(args.yaml)
        self.verbose = args.verbose
        self.predictor = Predictor(args.yaml)

    def evaluate(self):
        labels_test, X_test = load_data(self.config.data["test"])
        y_test: np.ndarray = self.predictor.label_encoder.transform(labels_test)
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
