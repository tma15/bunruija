import numpy as np
from scipy.sparse import csr_matrix

from bunruija.data import Dictionary


class SequenceVectorizer:
    def __init__(
            self,
            tokenizer,
            max_features=None,
            keep_raw_word=True,
            **kwargs):

        self.tokenizer = tokenizer
        self.dictionary = Dictionary()
        self.max_features = max_features
        self.keep_raw_word = keep_raw_word

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def fit(self, raw_documents):
        for row_id, document in enumerate(raw_documents):
            elements = self.tokenizer(document)
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

    def transform(self, raw_documents):
        data = []
        raw_words = []
        row = []
        col = []
        max_col = 0

        for row_id, document in enumerate(raw_documents):
            elements = self.tokenizer(document)
            max_col = max(max_col, len(elements))

            for i, element in enumerate(elements):
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

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        x = self.transform(raw_documents)
        return x

    def __call__(self, text):
        ret = self.tokenizer(text)
        return ret
