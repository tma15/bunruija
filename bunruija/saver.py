import pickle

from sklearn.preprocessing import LabelEncoder  # type: ignore

from .dataclass import BunruijaConfig


class Saver:
    def __init__(self, config: BunruijaConfig):
        self.config = config

    def __call__(self, model, label_encoder: LabelEncoder):
        if not self.config.output_dir.exists():
            self.config.output_dir.mkdir(parents=True)

        with open(self.config.output_dir / "model.bunruija", "wb") as f:
            model_data = {
                "pipeline": model,
                "label_encoder": label_encoder,
            }
            pickle.dump(model_data, f)
