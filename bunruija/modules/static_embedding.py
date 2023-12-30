import logging
import os

import torch

from bunruija.modules.vector_processor import PretrainedVectorProcessor  # type: ignore

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
        return (
            f"{self.__class__.__name__}"
            f"(vocab_size={self.length}, embedding_size={self.dim_emb})"
        )

    def forward(self, batch):
        bsz = len(batch["words"])
        max_seq_len = 0
        for batch_idx in range(len(batch["words"])):
            max_seq_len = max(max_seq_len, len(batch["words"][batch_idx]))

        embed = torch.zeros((bsz, max_seq_len, self.dim_emb))
        for batch_idx in range(len(batch["words"])):
            vec, status = self.processor.batch_query(tuple(batch["words"][batch_idx]))
            seq_len = len(batch["words"][batch_idx])
            embed[batch_idx, :seq_len] = torch.from_numpy(vec)

        return embed
