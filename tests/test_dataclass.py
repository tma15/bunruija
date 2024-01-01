import tempfile
from pathlib import Path
from unittest import TestCase

from bunruija import BunruijaConfig
from bunruija_cli import gen_yaml


class TestBunruijaConfig(TestCase):
    def test_from_yaml(self):
        with tempfile.TemporaryDirectory("test_config") as data_dir:
            yaml_file = str(Path(data_dir) / "test-binary.yaml")
            gen_yaml.main(
                [
                    "--model",
                    "sklearn.svm.SVC",
                    "-y",
                    yaml_file,
                ]
            )

            config = BunruijaConfig.from_yaml(yaml_file)
            self.assertTrue(hasattr(config, "data"))
            self.assertTrue(hasattr(config, "pipeline"))
            self.assertTrue(hasattr(config, "bin_dir"))

            self.assertEqual(
                config.pipeline[0].type,
                "sklearn.feature_extraction.text.TfidfVectorizer",
            )
            self.assertEqual(
                config.pipeline[0].args["tokenizer"]["type"],
                "bunruija.tokenizers.mecab_tokenizer.MeCabTokenizer",
            )
            self.assertEqual(config.pipeline[1].type, "sklearn.svm.SVC")
