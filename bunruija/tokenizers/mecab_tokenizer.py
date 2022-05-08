from typing import List

from fugashi import Tagger  # type: ignore

from ..filters import PosFilter
from .tokenizer import BaseTokenizer


class MeCabTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__(name="mecab")
        self.lemmatize = kwargs.get("lemmatize", False)
        exclude_pos = kwargs.get("exclude_pos", [])
        if len(exclude_pos) > 0:
            self.filters = [PosFilter(exclude_pos)]
        else:
            self.filters = None

        self.dict_path = kwargs.get("dict_path", None)
        if self.dict_path:
            self._mecab = Tagger(f"-d {self.dict_path}")
        else:
            self._mecab = Tagger()

    def __getstate__(self):
        return {
            "lemmatize": self.lemmatize,
            "filters": self.filters,
            "dict_path": self.dict_path,
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __getnewargs__(self):
        return ()

    def __reduce_ex__(self, proto):
        func = MeCabTokenizer
        args = self.__getnewargs__()
        state = self.__getstate__()
        listitems = None
        dictitems = None
        rv = (func, args, state, listitems, dictitems)
        return rv

    def __repr__(self) -> str:
        args = []
        args.append(f"lemmatize={self.lemmatize}")
        if self.dict_path:
            args.append(f"dict_path={self.dict_path}")
        if self.filters:
            args.append(f"filters={self.filters}")
        out = f'{self.__class__.__name__}({", ".join(args)})'
        return out

    def __call__(self, text: str) -> List[str]:
        ret = []
        for word in self._mecab(text):
            if self.filters and any(
                [f(word.surface, word.feature) for f in self.filters]
            ):
                continue
            else:
                if self.lemmatize:
                    if word.feature.lemma is None:
                        ret.append(word.surface)
                    else:
                        ret.append(word.feature.lemma)
                else:
                    ret.append(word.surface)
        return ret
