from typing import List, Optional


class BaseTokenizer:
    def __init__(self, name: Optional[str] = None):
        name = name

    def __call__(self, text: str) -> List[str]:
        raise NotImplementedError
