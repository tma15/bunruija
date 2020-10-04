
class BaseClassifier:
    def __init__(self):
        name = None

    def fit(self, X, y):
        raise NotImplementedError
