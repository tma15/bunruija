import random

import mmh3
import numpy as np
import torch

from bunruija.classifiers.classifier import NeuralBaseClassifier
from bunruija.classifiers.prado.string_projector import StringProjectorOp


class Hasher:
    def __init__(self, n_features=64):
        self.n_features = n_features

    def get_hash_codes(self, word):
        hash_ = mmh3.hash128(word)
        return hash_


class WeightMask:
    def __init__(self, index):
        self.index = index

    def __call__(self, module, _):
        mask = module.raw_weight.new_ones(module.raw_weight.size())
        if self.index.device != mask.device:
            mask.index_fill_(2, self.index.to(mask.device), 0.)
        else:
            mask.index_fill_(2, self.index, 0.)
        module.weight = module.raw_weight * mask


class ConvolutionLayer(torch.nn.Module):
    def __init__(self, kernel_size, dim_hid):
        super().__init__()
        self.dim_hid = dim_hid
        self.conv = torch.nn.Conv2d(
            1,
            self.dim_hid,
            kernel_size=(kernel_size, self.dim_hid),
            stride=1
        )

        self.batch_norm = torch.nn.BatchNorm1d(self.dim_hid)

    def __call__(self, x):
        x = self.conv(x).squeeze(3)
        x = self.batch_norm(x)
        return x


class ProjectAttentionLayer(torch.nn.Module):
    def __init__(self, kernel_size, dim_hid, skip_bigram=None):
        super().__init__()
        self.conv_value = ConvolutionLayer(kernel_size, dim_hid)

        if isinstance(skip_bigram, list):
            self.conv_value.conv.raw_weight = self.conv_value.conv.weight
            del self.conv_value.conv.weight
            weight_mask = WeightMask(torch.tensor(skip_bigram))
            self.conv_value.conv.register_forward_pre_hook(weight_mask)
        elif skip_bigram is None:
            pass
        else:
            raise ValueError(skip_gram)

        self.conv_attn = ConvolutionLayer(kernel_size, dim_hid)
        self.zero_pad = torch.nn.ZeroPad2d((0, 0, kernel_size - 1, 0))

    def __call__(self, x_value_in, x_attn_in):
        # (bsz, output_channel, seq_len)
        x_value_in = self.zero_pad(x_value_in)
        x_value = self.conv_value(x_value_in)

        # (bsz, output_channel, seq_len)
        x_attn_in = self.zero_pad(x_attn_in)
        x_attn = self.conv_attn(x_attn_in)
        x_attn = torch.softmax(x_attn, dim=2)

        x = x_attn * x_value
        # (bsz, output_channel)
        x = torch.sum(x, dim=2)
        return x


class ProjectedAttention(torch.nn.Module):
    def __init__(self, kernel_sizes, dim_hid, skip_bigrams=None):
        super().__init__()
        self.dim_hid = dim_hid

        if isinstance(skip_bigrams, list):
            self.layers = torch.nn.ModuleList([
                ProjectAttentionLayer(kernel_size, dim_hid, skip_bigram=skip_bigram)
                for kernel_size, skip_bigram in zip(kernel_sizes, skip_bigrams)
            ])
        else:
            self.layers = torch.nn.ModuleList([
                ProjectAttentionLayer(kernel_size, dim_hid, skip_bigram=None)
                for kernel_size in kernel_sizes
            ])

    def __call__(self, x_value_in, x_attn_in):
        x_list = []

        for layer in self.layers:
            x = layer(x_value_in, x_attn_in)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        return x


class PRADO(NeuralBaseClassifier):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.random_char = None
        self.make_fast = kwargs.get('make_fast', True)
        self.n_features = kwargs.get('n_features', 512)
        self.hasher = Hasher(self.n_features)
        self.distort = kwargs.get('distortion_probability', 0.25)

        if self.make_fast:
            self.string_proj = StringProjectorOp(self.n_features, self.distort)

        self.dim_emb = kwargs.get('dim_emb', 64)
        self.mapping_table = [0, 1, -1, 0]

        self.dim_hid = kwargs.get('dim_hid', 64)

        self.fc_value = torch.nn.Linear(self.n_features, self.dim_hid)
        self.fc_attn = torch.nn.Linear(self.n_features, self.dim_hid)

        dropout = kwargs.get('dropout', 0.5)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.kernel_sizes = kwargs.get('kernel_sizes', [2, 3, 3, 4])
        self.skip_bigrams = kwargs.get('skip_bigrams', [None, None, [1], [1, 2]])
        self.attention = ProjectedAttention(
            self.kernel_sizes,
            self.dim_hid,
            skip_bigrams=self.skip_bigrams)

        self.batch_norm_value = torch.nn.BatchNorm1d(self.dim_hid)
        self.batch_norm_attn = torch.nn.BatchNorm1d(self.dim_hid)

    def train(self, mode=True):
        super().train(mode)
        if self.make_fast:
            self.string_proj.train(mode)

    def init_layer(self, data):
        self.pad = 0
        self.fc = torch.nn.Linear(
            len(self.kernel_sizes) * self.dim_hid,
            len(self.labels),
            bias=True)

    def word_string_distort(self, word):
        if self.distort == 0 or len(word) == 0:
            return word
        else:
            if random.random() < self.distort:
                distortion_type = random.random()
                rindex = random.randint(0, len(word) - 1)
                if distortion_type < 0.33:
                    self.random_char = word[rindex]
                    word = word[:rindex] + word[rindex + 1:]
                elif distortion_type < 0.66:
                    if len(word) > 2:
                        self.random_char = word[rindex]
                        rindex2 = random.randint(0, len(word) - 1)
                        word = list(word)
                        word[rindex2] = word[rindex]
                        word = ''.join(word)
                elif self.random_char:
                    word = list(word)
                    word[rindex] = self.random_char
                    word = ''.join(word)
            return word

    def string_projection(self, batch):
        max_seq_len = max(len(words) for words in batch['words'])
        x = torch.zeros(
            len(batch['words']),
            max_seq_len,
            self.n_features,
            dtype=torch.float32)
        
        words_batch = batch['words']
        for batch_idx, words in enumerate(words_batch):
            for t, word in enumerate(words):
                if self.training:
                    word = self.word_string_distort(word)

                hash_ = self.hasher.get_hash_codes(word)
                projection = []

                b = self.mapping_table[hash_ & 0x3]
                projection.append(b)

                for i in range(1, self.n_features):
                    hash_ >>= 2
                    b = self.mapping_table[hash_ & 0x3]
                    projection.append(b)
                x[batch_idx, t] = torch.tensor(projection)
        return x

    def __call__(self, batch):
        if self.make_fast:
            projection = self.string_proj(batch['words'])
            projection = torch.from_numpy(projection)
        else:
            projection = self.string_projection(batch)

#         max_seq_len = max(len(words) for words in batch['words'])
#         for b in range(len(batch['words'])):
#             print(len(batch['words'][b]))
#             for w in range(max_seq_len):
#                 if w < len(batch['words'][b]):
#                     word = batch['words'][b][w]
#                     print(w, word, projection[b, w].tolist())
#                 else:
#                     print(w, projection[b, w].tolist())
#             print()
#         exit()

        projection = projection.to(batch['inputs'].device)
        mask = (batch['inputs'] == self.pad).unsqueeze(2)

        x_value = self.fc_value(projection)
        x_value = self.dropout(x_value)
        x_value = x_value.masked_fill_(mask, 0)
        x_value = self.batch_norm_value(x_value.transpose(1, 2)).transpose(1, 2)
        x_value = torch.relu(x_value)

        x_attn = self.fc_attn(projection)
        x_attn = self.dropout(x_attn)
        x_attn = x_attn.masked_fill_(mask, 0)
        x_attn = self.batch_norm_attn(x_attn.transpose(1, 2)).transpose(1, 2)
        x_attn = torch.relu(x_attn)

        x_value = x_value.unsqueeze(1)
        x_attn = x_attn.unsqueeze(1)

        x = self.attention(x_value, x_attn)
        x = self.fc(x)
        return x
