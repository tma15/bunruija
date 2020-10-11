import logging
import os

import numpy as np
from pymagnitude import converter, Magnitude
import torch


logger = logging.getLogger(__name__)


class StaticEmbedding(torch.nn.Module):
    def __init__(self, embedding_path):
        super().__init__()

        self.convert(embedding_path)
        self.magnitude = Magnitude(embedding_path + '.magnitude')
        self.dim_emb = self.magnitude.emb_dim

    def convert(self, embedding_path):
        ext = '.magnitude'
        output_file = embedding_path + ext

        if not os.path.exists(output_file):
            logger.info(f'Converting word vectorsto {output_file}')
            logger.info('This conversion will take a while')
            converter.convert(embedding_path,
                    output_file,
                    precision=7, subword=True,
                    subword_start=3,
                    subword_end=6,
                    approx=False, approx_trees=None)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.magnitude.length}, {self.dim_emb})'

    def __call__(self, batch):
        words = batch['words']

        bsz = len(batch['words'])
        seq_len = 0
        for batch_idx in range(len(batch['words'])):
            seq_len = max(seq_len, len(batch['words'][batch_idx]))

        embed = torch.zeros((bsz, seq_len, self.dim_emb))
        for batch_idx in range(len(batch['words'])):
            for t, word in enumerate(batch['words'][batch_idx]):
                if word in self.magnitude:
                    embed[batch_idx, t, :] = torch.from_numpy(
                        np.array(self.magnitude.query(word))
                    )
        return embed
