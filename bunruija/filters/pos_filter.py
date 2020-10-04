from .filter import BaseFilter


class PosFilter(BaseFilter):
    def __init__(self, exclude_pos=[]):
        super().__init__(
            name='pos'
        )
        self.exclude_pos = exclude_pos

    def __call__(self, surface, features, *args, **kwargs):
        for p in self.exclude_pos:
            if features[0] == p:
                return True
        return False
