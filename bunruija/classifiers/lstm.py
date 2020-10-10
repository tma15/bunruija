import logging

import numpy as np
import torch
import torch.nn.functional as F

from bunruija.feature_extraction.sequence import SequenceVectorizer


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


class LSTMClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        vectorizer = kwargs['vectorizer']
        label_encoder = kwargs['label_encoder']

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

        self.embed = torch.nn.Embedding(
            len(self.dictionary),
            self.dim_emb,
            padding_idx=self.pad,
        )

        self.lstm = torch.nn.LSTM(
#             input_size=embedding_size  self.key_vector.vector_size if embedding_path
#                 else embedding_size,
            self.dim_emb,
            hidden_size=self.dim_hid,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        self.out = torch.nn.Linear(
            2 * self.dim_hid,
            len(list(label_encoder.classes_)),
            bias=True)
        logger.info(self)

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

    def fit(self, X, y):
        data = self.convert_data(X, y)
        optimizer = torch.optim.Adam(self.parameters())

        for epoch in range(self.max_epochs):
            loss_epoch = 0.

            for batch in torch.utils.data.DataLoader(
                data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn(self.pad)
            ):
                optimizer.zero_grad()

                logits = self(batch)
                loss = F.nll_loss(
                    torch.log_softmax(logits, dim=1),
                    batch['labels']
                )
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()

            print(epoch, loss_epoch)

    def __call__(self, batch):
        x = batch['inputs']
        lengths = (x != self.pad).sum(dim=1)

        x = self.embed(x)
#         print(x.size())

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        rnn_out, (ht, ct) = self.lstm(packed)

        # (bsz, seq_len, 2 * hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
#         print(x.size())
        x = self.out(x[:, 0])
#         print(x.size())
        return x

    def predict(self, X):
        data = self.convert_data(X)

        y = []
        for batch in torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn(self.pad)
        ):
            maxi = torch.argmax(self(batch), dim=1)
            y.extend(maxi.tolist())
        return np.array(y)
