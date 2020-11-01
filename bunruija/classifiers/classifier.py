import logging
import time

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


def move_to_cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


class BaseClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        vectorizer = kwargs['vectorizer']
        if not isinstance(vectorizer, SequenceVectorizer):
            raise ValueError(vectorizer)

        self.kwargs = kwargs
        self.device = kwargs.get('device', 'cpu')
        self.max_epochs = kwargs.get('max_epochs', 3)
        self.batch_size = kwargs.get('batch_size', 20)

        self.dictionary = vectorizer.dictionary
        self.optimizer_type = kwargs.get('optimizer', 'adam')

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
                del loss

            elapsed = time.perf_counter() - start_at
            logger.info(f'epoch:{epoch} loss:{loss_epoch:.2f} elapsed:{elapsed:.2f}')

    def reset_module(self, **kwargs):
        pass

    def classifier_args(self):
        raise NotImplementedError

    def convert_data(self, X, y=None):
        raise NotImplementedError

    def build_optimizer(self):
        lr = float(self.kwargs.get('lr', 0.001))
        weight_decay = self.kwargs.get('weight_decay', 0.)

        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer_type}')
        return optimizer

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

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
