from pathlib import Path
import pickle
import yaml

import bunruija


class Predictor:
    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        with open(Path(config.get('bin_dir', '.')) / 'model.bunruija', 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['classifier']

            self.label_encoder = model_data['label_encoder']

    def __call__(self, text):
        x = [text]
        y = self.model.predict(x)

        if isinstance(text, str):
            label = self.label_encoder.inverse_transform(y)
            return label[0]
        else:
            return y
