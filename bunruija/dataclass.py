from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ruamel.yaml  # type: ignore


@dataclass
class PipelineUnit:
    type: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BunruijaConfig:
    data: dict[str, str]
    pipeline: list[PipelineUnit]
    output_dir: Path

    @classmethod
    def from_yaml(cls, config_file):
        with open(config_file) as f:
            yaml = ruamel.yaml.YAML()
            config = yaml.load(f)

            return cls(
                data=config["data"],
                pipeline=[PipelineUnit(**unit) for unit in config["pipeline"]],
                output_dir=Path(config.get("output_dir", "output")),
            )
