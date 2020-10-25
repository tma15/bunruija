import logging

import torch

from bunruija.classifiers.classifier import BaseClassifier
from bunruija.modules import StaticEmbedding


logger = logging.getLogger(__name__)


class LSTMClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        label_encoder = kwargs['label_encoder']
        self.embedding_path = kwargs.get('static_embedding_path', None)

        self.pad = self.dictionary.get_index('<pad>')
        self.dim_emb = kwargs.get('dim_emb', 32)
        self.dim_hid = kwargs.get('dim_hid', 128)
        self.num_layers = kwargs.get('num_layers', 1)
        self.bidirectional = kwargs.get('bidirectional', True)

        if self.embedding_path:
            self.static_embed = StaticEmbedding(self.embedding_path)
        else:
            self.static_embed = None

        self.embed = torch.nn.Embedding(
            len(self.dictionary),
            self.dim_emb,
            padding_idx=self.pad,
        )

        self.lstm = torch.nn.LSTM(
            input_size=(self.dim_emb + self.static_embed.dim_emb
                if self.static_embed else self.dim_emb),
            hidden_size=self.dim_hid,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        self.out = torch.nn.Linear(
            2 * self.dim_hid,
            len(list(label_encoder.classes_)),
            bias=True)
        logger.info(self)

    def reset_module(self, **kwargs):
        embedding_path = kwargs.get('static_embedding_path', None)
        if embedding_path:
            self.static_embed = StaticEmbedding(embedding_path)
        else:
            self.static_embed = None

    def classifier_args(self):
        return {
            'embedding_path': self.embedding_path
        }

    def convert_data(self, X, y=None):
        if isinstance(X, tuple):
            indices = X[0]
            raw_words = X[1]
            has_raw_words = True
        else:
            has_raw_words = False
            indices = X
            raw_words = None

        data = []
        for i in range(len(indices.indptr) - 1):
            start = indices.indptr[i]
            end = indices.indptr[i + 1]
            data_i = {
                'inputs': indices.data[start: end],
            }

            if y is not None:
                data_i['label'] = y[i]

            if has_raw_words:
                data_i['raw_words'] = raw_words[start: end]
            data.append(data_i)
        return data

    def __call__(self, batch):
        x = batch['inputs']
        lengths = (x != self.pad).sum(dim=1)

        x = self.embed(x)
        if self.static_embed is not None:
            x_static = self.static_embed(batch)
            x = torch.cat([x, x_static], dim=2)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        rnn_out, (ht, ct) = self.lstm(packed)

        # (bsz, seq_len, 2 * hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        x = self.out(x[:, 0])
        return x
