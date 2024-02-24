import numpy as np
from torch import FloatTensor, LongTensor
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .classifier import NeuralBaseClassifier


class TransformerClassifier(NeuralBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", None
        )

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.pad = tokenizer.pad_token_id

    def init_layer(self, data: list[dict]):
        y = []
        for data_i in data:
            y.append(data_i["label"])

        num_labels: int = len(np.unique(y))

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name_or_path, num_labels=num_labels
        )

    def forward(self, batch) -> FloatTensor:
        input_ids: LongTensor = batch["inputs"]
        attention_mask: LongTensor = batch["attention_mask"]
        output: SequenceClassifierOutput = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return output.logits
