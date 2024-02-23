from typing import Any, Callable

import numpy as np
import transformers  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.base import TransformerMixin  # type: ignore

from ..data import Dictionary
from ..tokenizers import MeCabTokenizer


class SequencePairVectorizer(TransformerMixin):
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] | None = None,
        max_features: int | None = None,
        dictionary: Dictionary = Dictionary(),
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.vocabulary_ = dictionary.index_to_element
        self.max_features = max_features

    @classmethod
    def module_name(cls) -> str:
        module_name = ".".join([cls.__module__, cls.__name__])
        return module_name

    def __repr__(self) -> str:
        args = []
        if self.tokenizer:
            args.append(f"tokenizer={self.tokenizer}")
        if self.max_features:
            args.append(f"max_features={self.max_features}")
        out = f'{self.__class__.__name__}({", ".join(args)})'
        return out

    def build_tokenizer(self) -> Callable[[str], list[str]]:
        if self.tokenizer is not None:
            return self.tokenizer

        self.tokenizer = MeCabTokenizer()
        return self.tokenizer

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_params(self, deep=True) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer,
            "max_features": self.max_features,
            "dictionary": self.dictionary,
        }

    def fit(self, raw_documents: list[list[str]], y=None) -> "SequencePairVectorizer":
        tokenizer = self.build_tokenizer()

        for row_id, document in enumerate(raw_documents):
            elements = tokenizer(document[0])
            for element in elements:
                self.dictionary.add(element)

        if self.max_features is not None:
            filtered_dict = Dictionary()
            for k, v in sorted(
                zip(self.dictionary.elements, self.dictionary.count),
                key=lambda x: x[1],
                reverse=True,
            )[: self.max_features]:
                filtered_dict.add(k, v)

            self.dictionary = filtered_dict
        return self

    def transform(
        self,
        raw_documents: list[list[str]],
    ) -> csr_matrix | tuple[csr_matrix, list[str]]:
        data = []
        row = []
        col = []
        max_col = 0

        tokenizer = self.build_tokenizer()
        for row_id, document in enumerate(raw_documents):
            if isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
                elements = tokenizer(
                    text=document[0], text_pair=document[1], truncation=True
                )
            else:
                elements_1 = tokenizer(document[0])
                elements_2 = tokenizer(document[1])
                elements = elements_1 + elements_2

            if isinstance(elements, transformers.tokenization_utils_base.BatchEncoding):
                input_ids = elements["input_ids"]
                max_col = max(max_col, len(input_ids))

                for i, index in enumerate(input_ids):
                    data.append(index)
                    row.append(row_id)
                    col.append(i)

            else:
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
        return s

    def __call__(self, text):
        tokenizer = self.build_tokenizer()
        ret = tokenizer(text)
        return ret
