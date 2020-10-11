import MeCab

from bunruija.tokenizers import BaseTokenizer
from bunruija.filters import PosFilter


class MeCabTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__(
            name='mecab'
        )
        self.lemmatize = kwargs.get('lemmatize', False)
        exclude_pos = kwargs.get('exclude_pos', [])
        self.filters = [PosFilter(p) for p in exclude_pos]

        dict_path = kwargs.get('dict_path', None)
        if dict_path:
            self._mecab = MeCab.Tagger(f'-d {dict_path}')
        else:
            self._mecab = MeCab.Tagger()

    def __call__(self, text):
        result = self._mecab.parse(text).rstrip()
        ret = []
        for line in result.splitlines()[:-1]:
            surface, feature = line.split('\t')
            features = feature.split(',')

            if any([f(surface, features) for f in self.filters]):
                continue
            else:
                if self.lemmatize:
                    ret.append(surface if features[6] == '*' else features[6])
                else:
                    ret.append(surface)
        return ret
