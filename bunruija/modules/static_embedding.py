import logging
import os

import torch

from bunruija.modules.vector_processor import PretrainedVectorProcessor


logger = logging.getLogger(__name__)


class StaticEmbedding(torch.nn.Module):
    def __init__(self, embedding_path):
        super().__init__()

        self.processor = PretrainedVectorProcessor()

        self.embedding_path = embedding_path
        self.convert(self.embedding_path)
        self.processor.load(self.embedding_path + ".bunruija")
        self.dim_emb = self.processor.emb_dim
        self.length = self.processor.length

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.processor.load(self.embedding_path + ".bunruija")

    def convert(self, embedding_path):
        ext = ".bunruija"
        output_file = embedding_path + ext

        if not os.path.exists(output_file):
            logger.info(f"Converting word vectors to {output_file}")
            logger.info("This conversion will take a while")
            self.processor.convert(output_file, embedding_path)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.length}, {self.dim_emb})"

    def __call__(self, batch):
        bsz = len(batch["words"])
        seq_len = 0
        for batch_idx in range(len(batch["words"])):
            seq_len = max(seq_len, len(batch["words"][batch_idx]))

        embed = torch.zeros((bsz, seq_len, self.dim_emb))
        for batch_idx in range(len(batch["words"])):
            for t, word in enumerate(batch["words"][batch_idx]):
                if word in self.processor:
                    vec, status = self.processor.query(word)
                    embed[batch_idx, t, :] = torch.from_numpy(vec)
        return embed
