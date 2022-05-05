from pathlib import Path
import pickle


class Saver:
    def __init__(self, config):
        self.config = config

    def __call__(self, model):
        with open(Path(self.config.get("bin_dir", ".")) / "model.bunruija", "rb") as f:
            model_data = pickle.load(f)

        with open(Path(self.config.get("bin_dir", ".")) / "model.bunruija", "wb") as f:
            model_data["classifier"] = model
            pickle.dump(model_data, f)
