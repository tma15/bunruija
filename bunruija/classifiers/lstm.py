import logging

import numpy as np
import torch

from bunruija.classifiers.classifier import NeuralBaseClassifier
from bunruija.modules import StaticEmbedding


logger = logging.getLogger(__name__)


class LSTMClassifier(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embedding_path = kwargs.get('static_embedding_path', None)

        self.dim_emb = kwargs.get('dim_emb', 256)
        self.dim_hid = kwargs.get('dim_hid', 512)
        self.dropout_prob = kwargs.get('dropout', 0.15)
        self.num_layers = kwargs.get('num_layers', 1)
        self.bidirectional = kwargs.get('bidirectional', True)

        if self.embedding_path:
            self.static_embed = StaticEmbedding(self.embedding_path)
        else:
            self.static_embed = None

        self.dropout = torch.nn.Dropout(self.dropout_prob)

        self.lstm = torch.nn.LSTM(
            input_size=(
                self.dim_emb + self.static_embed.dim_emb
                if self.static_embed else self.dim_emb
            ),
            hidden_size=self.dim_hid,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.
        )

    def init_layer(self, data):
        max_input_idx = 0
        for data_i in data:
            max_input_idx = max(max_input_idx, np.max(data_i['inputs']))

        self.embed = torch.nn.Embedding(
            max_input_idx + 1,
            self.dim_emb,
            padding_idx=0,
        )
        self.pad = 0

        self.out = torch.nn.Linear(
            2 * self.dim_hid,
            len(self.labels),
            bias=True)

    def __call__(self, batch):
        x = batch['inputs']
        lengths = (x != self.pad).sum(dim=1)

        if (x >= self.embed.weight.size(0)).any():
            logger.error(f'elements of x are larger than embedding size')
            logger.error(x)

        x = self.embed(x)
        if self.static_embed is not None:
            x_static = self.static_embed(batch)
            x_static = x_static.to(x.device)
            x = torch.cat([x, x_static], dim=2)

        x = self.dropout(x)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        rnn_out, (ht, ct) = self.lstm(packed)

        # (bsz, seq_len, 2 * hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        x = self.dropout(x)
        x = self.out(x[:, 0])
        return x
