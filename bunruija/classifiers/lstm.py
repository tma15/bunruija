import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from bunruija.feature_extraction.sequence import SequenceVectorizer
from bunruija.modules import StaticEmbedding


logger = logging.getLogger(__name__)


def collate_fn(padding_value):
    def _collate_fn(samples):

        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(sample['inputs']) for sample in samples],
            batch_first=True,
            padding_value=padding_value)

        batch = {'inputs': inputs}
        if 'label' in samples[0]:
            labels = torch.tensor([sample['label'] for sample in samples])
            batch['labels'] = labels

        if 'raw_words' in samples[0]:
            words = [sample['raw_words'] for sample in samples]
            batch['words'] = words
        return batch
    return _collate_fn


def move_to_cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


class LSTMClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs

        vectorizer = kwargs['vectorizer']
        label_encoder = kwargs['label_encoder']
        self.embedding_path = kwargs.get('static_embedding_path', None)
        self.device = kwargs.get('device', 'cpu')

        if not isinstance(vectorizer, SequenceVectorizer):
            raise ValueError(vectorizer)

        self.max_epochs = kwargs.get('max_epochs', 3)
        self.batch_size = kwargs.get('batch_size', 20)

        self.dictionary = vectorizer.dictionary
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

        self.optimizer_type = kwargs.get('optimizer', 'adam')

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

    def build_optimizer(self):
        lr = float(self.kwargs.get('lr', 0.001))
        weight_decay = self.kwargs.get('weight_decay', 0.)

        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def fit(self, X, y):
        self.to(self.device)

        self.train()

        data = self.convert_data(X, y)

        optimizer = self.build_optimizer()
        logger.info(f'{optimizer}')
        start_at = time.perf_counter()

        for epoch in range(self.max_epochs):
            loss_epoch = 0.

            for batch in torch.utils.data.DataLoader(
                data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn(self.pad)
            ):
                self.zero_grad()

                if self.device.startswith('cuda'):
                   batch = move_to_cuda(batch)

                logits = self(batch)
                loss = F.nll_loss(
                    torch.log_softmax(logits, dim=1),
                    batch['labels']
                )
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()

            elapsed = time.perf_counter() - start_at
            logger.info(f'epoch:{epoch} loss:{loss_epoch:.2f} elapsed:{elapsed:.2f}')

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

    def predict(self, X):
        self.eval()

        data = self.convert_data(X)

        y = []
        for batch in torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn(self.pad)
        ):
            if self.device.startswith('cuda'):
               batch = move_to_cuda(batch)
        
            maxi = torch.argmax(self(batch), dim=1)
            y.extend(maxi.tolist())
        return np.array(y)
