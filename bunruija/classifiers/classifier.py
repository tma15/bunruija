import time
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore

logger = getLogger(__name__)


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class Collator:
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, samples):
        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(sample["inputs"]) for sample in samples],
            batch_first=True,
            padding_value=self.padding_value,
        )

        attention_mask = inputs != self.padding_value
        batch = {"inputs": inputs, "attention_mask": attention_mask}
        if "label" in samples[0]:
            labels = torch.tensor([sample["label"] for sample in samples])
            batch["labels"] = labels

        if "raw_words" in samples[0]:
            words = [sample["raw_words"] for sample in samples]
            batch["words"] = words
        return batch


def move_to_cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


class NeuralBaseClassifier(BaseClassifier, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.device = kwargs.get("device", "cpu")
        self.max_epochs = kwargs.get("max_epochs", 3)
        self.batch_size = kwargs.get("batch_size", 20)
        self.num_workers = kwargs.get("num_workers", 1)

        self.log_interval = kwargs.get("log_interval", 100)
        self.optimizer_type = kwargs.get("optimizer", "adam")
        self.save_every_step = kwargs.get("save_every_step", -1)
        self.saver = kwargs.get("saver", None)
        self.labels = set()
        logger.info(f"device: {self.device}")

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def init_layer(self, data):
        raise NotImplementedError

    def convert_data(self, X, y=None):
        if y is not None:
            logger.info("Loading data")

        if len(X) == 2 and isinstance(X[1], list):
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
                "inputs": indices.data[start:end],
            }

            if y is not None:
                data_i["label"] = y[i]
                self.labels.add(y[i])

            if has_raw_words:
                data_i["raw_words"] = raw_words[start:end]
            data.append(data_i)
        return data

    def fit(self, X, y):
        data = self.convert_data(X, y)
        self.init_layer(data)

        optimizer = self.build_optimizer()
        logger.info(f"{optimizer}")
        start_at = time.perf_counter()

        self.to(self.device)
        self.train()

        logger.info(f"{self}")
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Num params: {num_parameters}")
        step = 0
        loss_accum = 0
        n_samples_accum = 0
        start_at_accum = time.perf_counter()

        collator = Collator(self.pad)
        for epoch in range(self.max_epochs):
            for batch in torch.utils.data.DataLoader(
                data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=self.num_workers,
            ):
                self.zero_grad()

                if self.device.startswith("cuda"):
                    batch = move_to_cuda(batch)

                logits = self(batch)

                loss = F.nll_loss(
                    torch.log_softmax(logits, dim=1),
                    batch["labels"],
                    reduction="sum",
                )
                loss_accum += loss.item()
                n_samples_accum += len(batch["labels"])
                (loss / len(batch["labels"])).backward()
                optimizer.step()
                step += 1
                del loss

                if step % self.log_interval == 0:
                    loss_accum /= n_samples_accum
                    elapsed = time.perf_counter() - start_at
                    batch_per_sec = n_samples_accum / (
                        time.perf_counter() - start_at_accum
                    )
                    logger.info(
                        f"epoch:{epoch+1} step:{step} "
                        f"loss:{loss_accum:.2f} bps:{batch_per_sec:.2f} "
                        f"elapsed:{elapsed:.2f}"
                    )
                    loss_accum = 0
                    n_samples_accum = 0
                    start_at_accum = time.perf_counter()

                if (
                    self.save_every_step > -1
                    and self.saver
                    and step % self.save_every_step == 0
                ):
                    self.saver(self)

    def reset_module(self, **kwargs):
        pass

    def classifier_args(self):
        raise NotImplementedError

    def build_optimizer(self):
        lr = float(self.kwargs.get("lr", 0.001))
        weight_decay = self.kwargs.get("weight_decay", 0.0)

        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        return optimizer

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def predict(self, X):
        self.to(self.device)
        self.eval()

        data = self.convert_data(X)

        y = []
        collator = Collator(self.pad)
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.num_workers,
        )
        for batch in data_loader:
            if self.device.startswith("cuda"):
                batch = move_to_cuda(batch)

            with torch.inference_mode():
                maxi = torch.argmax(self(batch), dim=1)
            y.extend(maxi.tolist())
        return np.array(y)
