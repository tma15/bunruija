import tempfile
import unittest
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.svm import SVC  # type: ignore

from bunruija import BunruijaConfig, PipelineBuilder
from bunruija_cli import gen_yaml


class TestPipelineBuilder(unittest.TestCase):
    def test_build(self):
        model = "sklearn.svm.SVC"
        with tempfile.TemporaryDirectory("test_pipeline") as data_dir:
            yaml_file = Path(data_dir) / "test-pipelinebuilder.yaml"
            gen_yaml.main(
                [
                    "--model",
                    model,
                    "-y",
                    str(yaml_file),
                ]
            )

            config = BunruijaConfig.from_yaml(yaml_file)
            builder = PipelineBuilder(config)
            model = builder.build()
            self.assertTrue(isinstance(model, Pipeline))

            self.assertEqual(
                model.steps[0][0], "sklearn.feature_extraction.text.TfidfVectorizer"
            )
            self.assertTrue(isinstance(model.steps[0][1], TfidfVectorizer))
            self.assertEqual(model.steps[1][0], "sklearn.svm.SVC")
            self.assertTrue(isinstance(model.steps[1][1], SVC))
