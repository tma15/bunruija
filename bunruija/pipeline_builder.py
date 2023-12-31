from logging import getLogger
from typing import List, Union

from sklearn.pipeline import Pipeline  # type: ignore

from .classifiers.classifier import NeuralBaseClassifier
from .dataclass import BunruijaConfig, PipelineUnit
from .registry import BUNRUIJA_REGISTRY
from .saver import Saver
from .tokenizers.util import build_tokenizer

logger = getLogger(__name__)


class PipelineBuilder:
    def __init__(self, config: BunruijaConfig):
        self.config = config

    def _maybe_build_tokenizer(self, pipeline_unit: PipelineUnit):
        """If a pipeline_unit has a tokenizer as an argument, build the tokenizer"""
        if "tokenizer" in pipeline_unit.args:
            tokenizer = build_tokenizer(pipeline_unit.args["tokenizer"])
            pipeline_unit.args["tokenizer"] = tokenizer

    def _maybe_build_saver(self, pipeline_unit: PipelineUnit):
        if isinstance(BUNRUIJA_REGISTRY[pipeline_unit.type], NeuralBaseClassifier):
            pipeline_unit.args["saver"] = self.saver

    def _maybe_update_arg(self, pipeline_unit: PipelineUnit):
        self._maybe_build_tokenizer(pipeline_unit)
        self._maybe_build_saver(pipeline_unit)

        def _update_arg_value(x):
            if isinstance(x, list):
                return [_update_arg_value(_) for _ in x]
            elif isinstance(x, dict):
                if "tokenizer" in x.get("args", {}):
                    tokenizer = build_tokenizer(x["args"]["tokenizer"])
                    x["args"]["tokenizer"] = tokenizer

                if x["type"] in BUNRUIJA_REGISTRY and isinstance(
                    BUNRUIJA_REGISTRY[x["type"]], NeuralBaseClassifier
                ):
                    x.args["saver"] = self.saver

                if x["type"].startswith("pipeline"):
                    estimator = self.build_estimator(
                        [PipelineUnit(**_) for _ in x.get("args", {})],
                        pipeline_idx=x["type"],
                    )
                else:
                    estimator = BUNRUIJA_REGISTRY[x["type"]](**x.get("args", {}))
                return estimator
            else:
                return x

        for key, value in pipeline_unit.args.items():
            pipeline_unit.args[key] = _update_arg_value(value)

            # Because some argments such as ngra_range in TfidfVectorizer assumes tuple
            # while YAML cannnot interpret tuple, convert list to tuple according to
            # their constraints.
            if (
                hasattr(BUNRUIJA_REGISTRY[pipeline_unit.type], "_parameter_constraints")
                and key in BUNRUIJA_REGISTRY[pipeline_unit.type]._parameter_constraints
            ):
                constraints = BUNRUIJA_REGISTRY[
                    pipeline_unit.type
                ]._parameter_constraints[key]
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
            estimator = BUNRUIJA_REGISTRY[pipeline_units.type](**pipeline_units.args)

        # Because Pipeline of scikit-learn requires the tuple of name and estimator,
        # this functions returns them
        return estimator_type, estimator

    def build(self):
        setting = self.config.pipeline
        self.saver = Saver(self.config)

        model = self.build_estimator(setting)[1]
        logger.info(model)
        return model
