import pickle

from .dataclass import BunruijaConfig


class Saver:
    def __init__(self, config: BunruijaConfig):
        self.config = config

    def __call__(self, model):
        with open(self.config.bin_dir / "model.bunruija", "rb") as f:
            model_data = pickle.load(f)

        with open(self.config.bin_dir / "model.bunruija", "wb") as f:
            model_data["pipeline"] = model
            pickle.dump(model_data, f)
