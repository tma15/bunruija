class BaseFilter:
    def __init__(self, name=None):
        self.name = name

    def __call__(self):
        raise NotImplementedError
