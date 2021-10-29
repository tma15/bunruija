
class BaseTokenizer:
    def __init__(self, name=None):
        name = name

    def __call__(self, text):
        raise NotImplementedError
