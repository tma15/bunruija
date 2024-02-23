from collections import UserDict
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
    label_column: str = "label"
    text_column: str | list[str] = "text"


@dataclass
class BunruijaConfig:
    pipeline: list[PipelineUnit]
    output_dir: Path
    data: DataConfig | None = None
    dataset_args: UserDict | None = None

    @classmethod
    def from_yaml(cls, config_file):
        with open(config_file) as f:
            yaml = ruamel.yaml.YAML()
            config = yaml.load(f)

            label_column: str = config["data"].pop("label_column", "label")
            text_column: str | list[str] = config["data"].pop("text_column", "text")

            return cls(
                data=DataConfig(label_column=label_column, text_column=text_column),
                pipeline=[PipelineUnit(**unit) for unit in config["pipeline"]],
                output_dir=Path(config.get("output_dir", "output")),
                dataset_args=UserDict(config["data"]["args"]),
            )
