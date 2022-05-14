import logging
from typing import Optional

import numpy as np
import torch
from torch.nn import Module

from bunruija.classifiers.classifier import NeuralBaseClassifier
from bunruija.modules import StaticEmbedding


logger = logging.getLogger(__name__)


class LSTMClassifier(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embedding_path: Optional[str] = kwargs.get("static_embedding_path", None)

        self.dim_emb: int = kwargs.get("dim_emb", 256)
        self.dim_hid: int = kwargs.get("dim_hid", 512)
        self.dropout_prob: float = kwargs.get("dropout", 0.15)
        self.num_layers: int = kwargs.get("num_layers", 1)
        self.bidirectional: bool = kwargs.get("bidirectional", True)

        if self.embedding_path:
            self.static_embed = StaticEmbedding(self.embedding_path)
        else:
            self.static_embed = None

        self.dropout: Module = torch.nn.Dropout(self.dropout_prob)

        self.lstm: Module = torch.nn.LSTM(
            input_size=(
                self.dim_emb + self.static_embed.dim_emb
                if self.static_embed
                else self.dim_emb
            ),
            hidden_size=self.dim_hid,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )

    def init_layer(self, data):
        max_input_idx = 0
        for data_i in data:
            max_input_idx = max(max_input_idx, np.max(data_i["inputs"]))

        self.embed = torch.nn.Embedding(
            max_input_idx + 1,
            self.dim_emb,
            padding_idx=0,
        )
        self.pad = 0

        self.fc1 = torch.nn.Linear(2 * self.dim_hid, self.dim_hid, bias=True)
        self.activation_fn = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(self.dim_hid, len(self.labels), bias=True)

    def forward(self, batch):
        x = batch["inputs"]
        bsz = x.size(0)
        lengths = (x != self.pad).sum(dim=1)

        x = self.embed(x)
        if self.static_embed is not None:
            x_static = self.static_embed(batch)
            x_static = x_static.to(x.device)
            x = torch.cat([x, x_static], dim=2)

        x = self.dropout(x)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (ht, _) = self.lstm(packed)

        ht = ht.view(2, self.num_layers, bsz, self.dim_hid)
        x = torch.cat(
            [ht[0, self.num_layers - 1, :, :], ht[1, self.num_layers - 1, :, :]], dim=1
        )
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x
