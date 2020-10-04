from sklearn.feature_extraction.text import TfidfVectorizer

from bunruija import options
from bunruija.data import Dictionary
from bunruija.tokenizers import MeCabTokenizer
from bunruija.filters import PosFilter
from bunruija import SequenceVectorizer


def main():
    parser = options.get_default_preprocessing_parser()
    args = parser.parse_args()
    print(args)


    texts = [
        'すももももももものうち',
        '昨日、ご飯を食べに行った',
        '明日は雨かもしれない',
    ]

    v = TfidfVectorizer(
        tokenizer=MeCabTokenizer(
            lemmatize=True,
            filters=[
                PosFilter(exclude_pos=['助詞'])
            ]
        ),
    )
    x = v.fit_transform(texts)
    print(x)

    v = SequenceVectorizer(
        tokenizer=MeCabTokenizer(
            lemmatize=False,
        ),
    )
    x = v.fit_transform(texts)
    print(x)
