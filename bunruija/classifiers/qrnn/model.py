import numpy as np
import torch

from bunruija.classifiers.classifier import NeuralBaseClassifier


class QRNNLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, window_size=2, bidirectional=True):
        super().__init__()

        self.num_gates = 3
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.fc = torch.nn.Linear(
                self.window_size * input_size, 2 * output_size * self.num_gates
            )
        else:
            self.fc = torch.nn.Linear(
                self.window_size * input_size, output_size * self.num_gates
            )

    def forward(self, x):
        bsz = x.size(0)
        seq_len = x.size(1)
        window_tokens = [x]
        for i in range(self.window_size - 1):
            prev_x = x[:, : -(i + 1), :]
            prev_x = torch.cat(
                [prev_x.new_zeros(bsz, i + 1, self.input_size), prev_x], dim=1
            )
            window_tokens.insert(0, prev_x)
        x = torch.stack(window_tokens, dim=2)
        x = x.view(bsz, seq_len, -1)
        x = self.fc(x)
        z, f, o = x.chunk(self.num_gates, dim=2)

        z = torch.tanh(z)
        f = torch.sigmoid(f)
        seq_len = z.size(1)

        c = torch.zeros_like(z)

        if self.bidirectional:
            c = c.view(bsz, seq_len, 2, self.output_size)
            f = f.view(bsz, seq_len, 2, self.output_size)
            z = z.view(bsz, seq_len, 2, self.output_size)
            for t in range(seq_len):
                if t == 0:
                    c[:, t, 0] = (1 - f[:, t, 0]) * z[:, t, 0]
                else:
                    c[:, t, 0] = (
                        f[:, t, 0] * c[:, t - 1, 0].clone()
                        + (1 - f[:, t, 0]) * z[:, t, 0]
                    )
            for t in range(seq_len - 1, -1, -1):
                if t == seq_len - 1:
                    c[:, t, 0] = (1 - f[:, t, 0]) * z[:, t, 0]
                else:
                    c[:, t, 0] = (
                        f[:, t, 0] * c[:, t + 1, 0].clone()
                        + (1 - f[:, t, 0]) * z[:, t, 0]
                    )
            c = c.view(bsz, seq_len, 2 * self.output_size)
        else:
            for t in range(seq_len):
                if t == 0:
                    c[:, t] = (1 - f[:, t]) * z[:, t]
                else:
                    c[:, t] = f[:, t] * c[:, t - 1].clone() + (1 - f[:, t]) * z[:, t]

        h = torch.sigmoid(o) * c
        return h


class QRNN(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim_emb = kwargs.get("dim_emb", 256)
        self.dim_hid = kwargs.get("dim_hid", 128)
        self.dropout_prob = kwargs.get("dropout", 0.15)
        self.window_size = kwargs.get("window_size", 3)

        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.layers = torch.nn.ModuleList()
        self.bidirectional = kwargs.get("bidirectional", True)
        num_layers = kwargs.get("num_layers", 2)
        for i in range(num_layers):
            if i == 0:
                input_size = self.dim_emb
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

    def __call__(self, batch):
        src_tokens = batch["inputs"]
        lengths = (src_tokens != self.pad).sum(dim=1)

        x = self.embed(src_tokens)
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
