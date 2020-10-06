import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from bunruija.tokenizers import build_tokenizer
from bunruija.feature_extractor import SequenceVectorizer


class Binarizer:
    def __init__(self, config_file):
        self.config_file = config_file

    def load_data(self, input_file):
        labels = [
            'ごはん',
            'ごはん',
            '天気',
        ]
        texts = [
            'すももももももものうち',
            '昨日、ご飯を食べに行った',
            '明日は雨かもしれない',
        ]
        return labels, texts

    def binarize(self, input_file):
        labels, texts = self.load_data(input_file)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)

        v = TfidfVectorizer(
            tokenizer=build_tokenizer()
        )

        x = v.fit_transform(texts)

        with open('preprocessor.bunruija', 'wb') as f:
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
                'label': y,
                'data': x,
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
