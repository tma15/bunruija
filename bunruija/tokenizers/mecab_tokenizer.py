from bunruija.tokenizers import BaseTokenizer

import MeCab


class MeCabTokenizer(BaseTokenizer):
    def __init__(self, lemmatize=False, filters=[]):
        super().__init__(
            name='mecab'
        )
        self._mecab = MeCab.Tagger()
        self.lemmatize = lemmatize
        self.filters = filters

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
#         print(ret)
        return ret
