from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ruamel.yaml  # type: ignore


@dataclass
class PipelineUnit:
    type: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    train: Path = field(default_factory=Path)
    dev: Path = field(default_factory=Path)
    test: Path = field(default_factory=Path)
    label_column: str = "label"
    text_column: str = "text"

    def __post_init__(self):
        self.train = Path(self.train)
        self.dev = Path(self.dev)
        self.test = Path(self.test)


@dataclass
class BunruijaConfig:
    data: DataConfig
    pipeline: list[PipelineUnit]
    output_dir: Path

    @classmethod
    def from_yaml(cls, config_file):
        with open(config_file) as f:
            yaml = ruamel.yaml.YAML()
            config = yaml.load(f)

            return cls(
                data=DataConfig(**config["data"]),
                pipeline=[PipelineUnit(**unit) for unit in config["pipeline"]],
                output_dir=Path(config.get("output_dir", "output")),
            )
