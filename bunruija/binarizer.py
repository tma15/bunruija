import csv
import os
from pathlib import Path
import pickle
import yaml

from sklearn.preprocessing import LabelEncoder

from bunruija.feature_extraction import build_vectorizer


class Binarizer:
    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        if not os.path.exists(self.config.get('bin_dir', '.')):
            os.makedirs(self.config.get('bin_dir', '.'))

    def load_data(self, data_path):
        labels = []
        texts = []
        with open(data_path) as f:
            reader = csv.reader(f)
            for row in reader:
                labels.append(row[0])
                texts.append(row[1])
        return labels, texts

    def binarize(self):
        labels_train, texts_train = self.load_data(self.config['preprocess']['data']['train'])

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(labels_train)

        v = build_vectorizer(self.config)
        x_train = v.fit_transform(texts_train)

        if 'dev' in self.config['preprocess']['data']:
            labels_dev, texts_dev = self.load_data(self.config['preprocess']['data']['dev'])
            y_dev = label_encoder.transform(labels_dev)
            x_dev = v.transform(texts_dev)

        if 'test' in self.config['preprocess']['data']:
            labels_test, texts_test = self.load_data(self.config['preprocess']['data']['test'])
            y_test = label_encoder.transform(labels_test)
            x_test = v.transform(texts_test)

        with open(Path(self.config.get('bin_dir', '.')) / 'model.bunruija', 'wb') as f:
            if v.tokenizer is None:
                tokenizer_name = None
            else:
                tokenizer_name = v.tokenizer.__class__.__name__

            v.set_params(tokenizer=None)
            pickle.dump({
                    'label_encoder': label_encoder,
                    'vectorizer': v,
                    'tokenizer': tokenizer_name
                }, f)

        with open(Path(self.config.get('bin_dir', '.')) / 'data.bunruija', 'wb') as f:
            data = {
                'label_train': y_train,
                'data_train': x_train,
            }
            
            if 'dev' in self.config['preprocess']['data']:
                data['label_dev'] = y_dev
                data['data_dev'] = x_dev

            if 'test' in self.config['preprocess']['data']:
                data['label_test'] = y_test
                data['data_test'] = x_test

            pickle.dump(data, f)
