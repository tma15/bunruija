import csv
import os
from pathlib import Path
import pickle
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from bunruija.tokenizers import build_tokenizer
from bunruija.feature_extractor import SequenceVectorizer


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
        labels_dev, texts_dev = self.load_data(self.config['preprocess']['data']['dev'])
        labels_test, texts_test = self.load_data(self.config['preprocess']['data']['test'])

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(labels_train)
        y_dev = label_encoder.transform(labels_dev)
        y_test = label_encoder.transform(labels_test)

        v = TfidfVectorizer(
            tokenizer=build_tokenizer(self.config)
        )

        x_train = v.fit_transform(texts_train)
        x_dev = v.transform(texts_dev)
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
            pickle.dump({
                'label_train': y_train,
                'data_train': x_train,
                'label_dev': y_dev,
                'data_dev': x_dev,
                'label_test': y_test,
                'data_test': x_test
            }, f)

#         v = SequenceVectorizer(
#             tokenizer=MeCabTokenizer(
#                 lemmatize=False,
#             ),
#         )
#         x = v.fit_transform(texts)
#         print(x)

#         with open('vectorizer.bj', 'wb') as f:
#             v.set_params(tokenizer=None)
#             pickle.dump(v, f)
