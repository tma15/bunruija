from typing import Optional

import numpy as np
import torch

from bunruija.classifiers.classifier import NeuralBaseClassifier
from bunruija.classifiers.qrnn.qrnn_layer import QRNNLayer
from bunruija.modules import StaticEmbedding


class QRNN(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embedding_path: Optional[str] = kwargs.get("static_embedding_path", None)

        self.dim_emb: int = kwargs.get("dim_emb", 256)
        self.dim_hid: int = kwargs.get("dim_hid", 128)
        self.dropout_prob: float = kwargs.get("dropout", 0.15)
        self.window_size: int = kwargs.get("window_size", 3)
        self.bidirectional: bool = kwargs.get("bidirectional", True)

        if self.embedding_path:
            self.static_embed = StaticEmbedding(self.embedding_path)
        else:
            self.static_embed = None

        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.layers = torch.nn.ModuleList()
        num_layers = kwargs.get("num_layers", 2)
        for i in range(num_layers):
            if i == 0:
                input_size = (
                    self.dim_emb + self.static_embed.dim_emb
                    if self.static_embed
                    else self.dim
                )
            else:
                input_size = 2 * self.dim_hid if self.bidirectional else self.dim_hid

            self.layers.append(
                QRNNLayer(
                    input_size,
                    self.dim_hid,
                    window_size=self.window_size,
                    bidirectional=self.bidirectional,
                )
            )

    def init_layer(self, data):
        self.pad = 0
        max_input_idx = 0
        for data_i in data:
            max_input_idx = max(max_input_idx, np.max(data_i["inputs"]))

        self.embed = torch.nn.Embedding(
            max_input_idx + 1,
            self.dim_emb,
            padding_idx=0,
        )

        self.out = torch.nn.Linear(
            2 * self.dim_hid if self.bidirectional else self.dim_hid,
            len(self.labels),
            bias=True,
        )

    def forward(self, batch):
        src_tokens = batch["inputs"]

        x = self.embed(src_tokens)
        if self.static_embed is not None:
            x_static = self.static_embed(batch)
            x_static = x_static.to(x.device)
            x = torch.cat([x, x_static], dim=2)

        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(x)

        x = torch.nn.functional.adaptive_max_pool2d(
            x, (1, 2 * self.dim_hid if self.bidirectional else self.dim_hid)
        )
        x = x.squeeze(1)
        x = self.out(x)
        return x
