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
            self.config = yaml.load(f)
        print(self.config)

    def load_data(self, loader=None):
        if loader:
            labels, texts = loader()
        else:
            raise NotImplementedError()
        return labels, texts

    def split_data(self, labels, data, train_dev_test_split=None):
        if train_dev_test_split:
            return train_dev_test_split(labels, data)
        else:
            raise NotImplementedError()

    def binarize(self, loader=None, train_dev_test_split=None):
        labels, texts = self.load_data(loader=loader)

        labels_train, texts_train, labels_dev, texts_dev, labels_test, texts_test = \
            self.split_data(labels, texts, train_dev_test_split=train_dev_test_split)
        print(f'train size: {len(labels_train)}')
        print(f'dev size: {len(labels_dev)}')
        print(f'test size: {len(labels_test)}')

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(labels_train)
        y_dev = label_encoder.transform(labels_dev)
        y_test = label_encoder.transform(labels_test)

        v = TfidfVectorizer(
            tokenizer=build_tokenizer()
        )

        x_train = v.fit_transform(texts_train)
        x_dev = v.transform(texts_dev)
        x_test = v.transform(texts_test)

        with open('model.bunruija', 'wb') as f:
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

        with open('data.bunruija', 'wb') as f:
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
