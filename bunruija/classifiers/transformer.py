import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoTokenizer

from bunruija.classifiers.classifier import NeuralBaseClassifier


class TransformerClassifier(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from_pretrained = kwargs.pop('from_pretrained', None)
        self.model = AutoModel.from_pretrained(from_pretrained)
        self.dropout = torch.nn.Dropout(kwargs.get('dropout', 0.1))

        tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        self.pad = tokenizer.pad_token_id

    def init_layer(self, data):
        y = []
        max_input_idx = 0
        for data_i in data:
            y.append(data_i['label'])

        num_classes = np.unique(y)
        self.out = torch.nn.Linear(
            self.model.config.hidden_size,
            len(num_classes),
            bias=True)

    def __call__(self, batch):
        input_ids = batch['inputs']
        x = self.model(input_ids)
        pooled_output = x[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.out(pooled_output)
        return logits
