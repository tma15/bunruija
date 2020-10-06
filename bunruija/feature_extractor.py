import numpy as np
from scipy.sparse import csr_matrix

from bunruija.data import Dictionary


class SequenceVectorizer:
    def __init__(self, tokenizer, max_features=None):
        self.tokenizer = tokenizer
        self.dictionary = Dictionary()
        self.max_features = max_features

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def fit(self, raw_documents):
        for row_id, document in enumerate(raw_documents):
            elements = self.tokenizer(document)
            for element in elements:
                self.dictionary.add(element)

    def fit_transform(self, raw_documents):
        data = []
        row = []
        col = []
        max_col = 0

        self.fit(raw_documents)

        for row_id, document in enumerate(raw_documents):
            elements = self.tokenizer(document)
            max_col = max(max_col, len(elements))

            for i, element in enumerate(elements):
                if element in self.dictionary:
                    index = self.dictionary.get_index(element)
                    data.append(index)
                    row.append(row_id)
                    col.append(i)

        data = np.array(data)
        row = np.array(row)
        col = np.array(col)

        s = csr_matrix((data, (row, col)), shape=(len(raw_documents), max_col))

        for i in range(len(s.indptr) - 1):
            start = s.indptr[i]
            end = s.indptr[i + 1]
            print(start, end, s.data[start: end])
        return s

    def __call__(self, text):
        ret = self.tokenizer(text)
        return ret
