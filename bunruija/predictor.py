import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder  # type: ignore

from . import BunruijaConfig


class Predictor:
    """Predicts labels"""

    def __init__(self, config_file):
        config = BunruijaConfig.from_yaml(config_file)
        model_path: Path = config.output_dir / "model.bunruija"

        with open(model_path, "rb") as f:
            model_data: dict = pickle.load(f)

            self.model = model_data["pipeline"]
            self.label_encoder: LabelEncoder = model_data["label_encoder"]

    def __call__(
        self,
        text: list[str],
        return_label_type: str = "id",
    ) -> np.ndarray | list[str]:
        y: np.ndarray = self.model.predict(text)

        if return_label_type == "str":
            label = self.label_encoder.inverse_transform(y)
        elif return_label_type == "id":
            label = y
        else:
            raise ValueError(return_label_type)
        return label
