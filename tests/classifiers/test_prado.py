import unittest

import torch

from bunruija.classifiers.prado.model import PRADO, StringProjector


class TestStringProjection(unittest.TestCase):
    def setUp(self):
        self.n_features = 4
        self.words = [["hello", "word", "fox"], ["hell1o", "world", "fox"]]

    def test_python(self):
        proj = StringProjector(self.n_features, 0.0)
        x = proj(self.words)
        print("python", x)
        print(x[0, 1], x[1, 2])
        print(x[0, 0], x[1, 0])


class TestPRADO(unittest.TestCase):
    def setUp(self):
        self.model = PRADO()

    def test_forward(self):
        data = [
            {"label": "fruits"},
            {"label": "sports"},
        ]
        self.model.init_layer(data)

        batch = {
            "inputs": torch.tensor(
                [
                    [1, 1, 1],
                    [2, 3, 0],
                ],
                dtype=torch.long,
            ),
            "words": [["apple", "banana", "ball"], ["soccer", "baseball"]],
        }
        self.model(batch)
