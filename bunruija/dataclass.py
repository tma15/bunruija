from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import ruamel.yaml  # type: ignore


@dataclass
class PipelineUnit:
    type: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BunruijaConfig:
    data: Dict[str, str]
    pipeline: List[PipelineUnit]
    bin_dir: Path

    @classmethod
    def from_yaml(cls, config_file):
        with open(config_file) as f:
            yaml = ruamel.yaml.YAML()
            config = yaml.load(f)

            return cls(
                data=config["data"],
                pipeline=[PipelineUnit(**unit) for unit in config["pipeline"]],
                bin_dir=Path(config.get("bin_dir", ".")),
            )
