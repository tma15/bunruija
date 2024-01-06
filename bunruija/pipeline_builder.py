import importlib
from logging import getLogger
from typing import List, Union

from sklearn.pipeline import Pipeline  # type: ignore
from transformers import AutoTokenizer, PreTrainedTokenizer

from .dataclass import BunruijaConfig, PipelineUnit
from .saver import Saver

logger = getLogger(__name__)


class PipelineBuilder:
    def __init__(self, config: BunruijaConfig):
        self.config = config

    def _load_tokenizer(self, tokenizer_config: dict):
        name = tokenizer_config["type"]
        module_elems: list[str] = name.split(".")
        module_name: str = ".".join(module_elems[:-1])
        cls_name: str = module_elems[-1]
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        tokenizer_args = tokenizer_config.get("args", {})
        if cls is AutoTokenizer or issubclass(cls, PreTrainedTokenizer):
            tokenizer = cls.from_pretrained(**tokenizer_args)
        else:
            tokenizer = cls(**tokenizer_args)

        return tokenizer

    def _load_class(self, name: str):
        module_elems: list[str] = name.split(".")
        module_name: str = ".".join(module_elems[:-1])
        cls_name: str = module_elems[-1]
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls

    def _maybe_build_tokenizer(self, pipeline_unit: PipelineUnit):
        """If a pipeline_unit has a tokenizer as an argument, build the tokenizer"""
        if "tokenizer" in pipeline_unit.args:
            tokenizer = self._load_tokenizer(pipeline_unit.args["tokenizer"])
            pipeline_unit.args["tokenizer"] = tokenizer

    def _maybe_update_arg(self, pipeline_unit: PipelineUnit):
        self._maybe_build_tokenizer(pipeline_unit)

        def _update_arg_value(x):
            if isinstance(x, list):
                return [_update_arg_value(_) for _ in x]
            elif isinstance(x, dict):
                if "tokenizer" in x.get("args", {}):
                    tokenizer = self._load_tokenizer(x["args"]["tokenizer"])
                    x["args"]["tokenizer"] = tokenizer

                # If type of x starts with pipeline, args of x is assumed to be list,
                # which is basically the list of weak learners in an ensemble model.
                if x["type"].startswith("pipeline"):
                    estimator = self.build_estimator(
                        [PipelineUnit(**_) for _ in x.get("args", [])],
                        pipeline_idx=x["type"],
                    )
                else:
                    estimator = self.build_estimator(PipelineUnit(**x))[1]
                return estimator
            else:
                return x

        for key, value in pipeline_unit.args.items():
            pipeline_unit.args[key] = _update_arg_value(value)
            cls = self._load_class(pipeline_unit.type)

            # Because some argments such as ngram_range in TfidfVectorizer assume tuple
            # while YAML cannot interpret as tuple, convert list to tuple according to
            # their constraints.
            if (
                hasattr(cls, "_parameter_constraints")
                and key in cls._parameter_constraints
            ):
                constraints = cls._parameter_constraints[key]
                if tuple in constraints and isinstance(value, list):
                    pipeline_unit.args[key] = tuple(pipeline_unit.args[key])

    def build_estimator(
        self,
        pipeline_units: Union[PipelineUnit, List[PipelineUnit]],
        pipeline_idx="pipeline",
    ):
        if isinstance(pipeline_units, list):
            estimators = [self.build_estimator(u) for u in pipeline_units]
            estimator_type = pipeline_idx
            memory = self.config.bin_dir / "cache"
            estimator = Pipeline(estimators, memory=str(memory))
        else:
            self._maybe_update_arg(pipeline_units)
            estimator_type = pipeline_units.type
            cls = self._load_class(pipeline_units.type)
            estimator = cls(**pipeline_units.args)

        # Because Pipeline of scikit-learn requires the tuple of name and estimator,
        # this functions returns them
        return estimator_type, estimator

    def build(self):
        setting = self.config.pipeline
        self.saver = Saver(self.config)

        model = self.build_estimator(setting)[1]
        logger.info(model)
        return model
