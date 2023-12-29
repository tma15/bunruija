import functools
import unittest

import torch

from bunruija.classifiers.qrnn.qrnn_layer import QRNNLayer


class TestQNNLayer(unittest.TestCase):
    def setUp(self):
        self.dim_emb = 2
        self.dim_hid = 16
        self.bidirectional = True
        self.layer = QRNNLayer(
            self.dim_emb,
            self.dim_hid,
            window_size=3,
            bidirectional=self.bidirectional,
        )

    def test_stride(self):
        bsz = 2
        seq_len = 4
        x = (torch.arange(bsz * seq_len * self.dim_emb) + 1).view(
            bsz, seq_len, self.dim_emb
        )
        x = self.layer.stride(x)

        expected = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 1, 2],
                    [0, 0, 1, 2, 3, 4],
                    [1, 2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7, 8],
                ],
                [
                    [0, 0, 0, 0, 9, 10],
                    [0, 0, 9, 10, 11, 12],
                    [9, 10, 11, 12, 13, 14],
                    [11, 12, 13, 14, 15, 16],
                ],
            ]
        )

        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(expected, x)

    def test_packed_sequence_forward(self):
        x = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 0],
            ]
        )
        lengths = (x != 0).sum(1)

        embedding = torch.nn.Embedding(10, self.dim_emb, padding_idx=0)
        x = embedding(x)
        print(x.size())

        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        print(x)
        x = self.layer(x)

    def test_forward(self):
        bsz = 3
        seq_len = 5
        x = torch.randn(bsz, seq_len, self.dim_emb)
        x = self.layer(x)

        hidden_size = self.dim_hid if not self.bidirectional else self.dim_hid * 2
        self.assertEqual(x.shape, (bsz, seq_len, hidden_size))
