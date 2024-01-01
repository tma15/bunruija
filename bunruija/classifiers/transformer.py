import numpy as np
import torch
from transformers import AutoModel  # type: ignore
from transformers import AutoTokenizer

from .classifier import NeuralBaseClassifier


class TransformerClassifier(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", None
        )
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.1))

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.pad = tokenizer.pad_token_id

    def init_layer(self, data):
        y = []
        for data_i in data:
            y.append(data_i["label"])

        num_classes = np.unique(y)
        self.out = torch.nn.Linear(
            self.model.config.hidden_size, len(num_classes), bias=True
        )

    def forward(self, batch):
        input_ids = batch["inputs"]
        attention_mask = batch["attention_mask"]
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = x[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.out(pooled_output)
        return logits
