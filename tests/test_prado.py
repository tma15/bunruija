import unittest

import torch

from bunruija.classifiers import PRADO


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
