import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
import transformers

from bunruija import tokenizers
from bunruija.data import Dictionary


class SequenceVectorizer(TransformerMixin):
    def __init__(
            self,
            tokenizer=None,
            max_features=None,
            keep_raw_word=True,
            only_raw_word=False,
            dictionary=Dictionary(),
            **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.vocabulary_ = dictionary.index_to_element
        self.max_features = max_features
        self.keep_raw_word = keep_raw_word
        self.only_raw_word = only_raw_word

    def __repr__(self):
        args = []
        if self.tokenizer:
            args.append(f'tokenizer={self.tokenizer}')
        if self.max_features:
            args.append(f'max_features={self.max_features}')
        args.append(f'keep_raw_word={self.keep_raw_word}')
        args.append(f'only_raw_word={self.only_raw_word}')
        out = f'{self.__class__.__name__}({", ".join(args)})'
        return out

    def build_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        
        self.tokenizer = tokenizers.build_default_tokenizer()
        return self.tokenizer

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_params(self, deep=True):
        return {
            'tokenizer': self.tokenizer,
            'max_features': self.max_features,
            'dictionary': self.dictionary,
            'keep_raw_word': self.keep_raw_word,
            'only_raw_word': self.only_raw_word,
        }

    def fit(self, raw_documents, y=None):
        if self.only_raw_word:
            return self

        tokenizer = self.build_tokenizer()

        for row_id, document in enumerate(raw_documents):
            elements = tokenizer(document)
            for element in elements:
                self.dictionary.add(element)

        if self.max_features is not None:
            filtered_dict = Dictionary()
            for k, v in sorted(
                zip(self.dictionary.elements, self.dictionary.count),
                key=lambda x: x[1], reverse=True
            )[:self.max_features]:

                filtered_dict.add(k, v)

            self.dictionary = filtered_dict
        return self

    def transform(self, raw_documents):
        data = []
        raw_words = []
        row = []
        col = []
        max_col = 0

        tokenizer = self.build_tokenizer()
        for row_id, document in enumerate(raw_documents):
            elements = tokenizer(document)

            if isinstance(elements, transformers.tokenization_utils_base.BatchEncoding):
                input_ids = elements['input_ids']
                max_col = max(max_col, len(input_ids))

                for i, index in enumerate(input_ids):
                    data.append(index)
                    row.append(row_id)
                    col.append(i)

            else:
                max_col = max(max_col, len(elements))

                for i, element in enumerate(elements):
                    if self.only_raw_word:
                        raw_words.append(element)
                        index = 1
                        data.append(index)
                        row.append(row_id)
                        col.append(i)
                    else:
                        if element in self.dictionary:
                            if self.keep_raw_word:
                                raw_words.append(element)

                            index = self.dictionary.get_index(element)

                            data.append(index)
                            row.append(row_id)
                            col.append(i)

        data = np.array(data)
        row = np.array(row)
        col = np.array(col)

        s = csr_matrix((data, (row, col)), shape=(len(raw_documents), max_col))
        if self.keep_raw_word:
            return s, raw_words
        else:
            return s

    def __call__(self, text):
        tokenizer = self.build_tokenizer()
        ret = tokenizer(text)
        return ret
