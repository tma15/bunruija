from typing import List, Optional


class BaseTokenizer:
    def __init__(self, name: Optional[str] = None):
        if name:
            self.name = name

    @classmethod
    def module_name(cls) -> str:
        module_name = ".".join([cls.__module__, cls.__name__])
        return module_name

    def __call__(self, text: str) -> List[str]:
        raise NotImplementedError
