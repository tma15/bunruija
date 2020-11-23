from pathlib import Path
import pickle
import random
import tempfile
import unittest

import numpy
import torch

from bunruija.modules import StaticEmbedding


def create_dummy_data(data_file, num_tokens=50, dim=100):
    with open(data_file, 'w') as f:
        data = torch.rand(num_tokens * dim)
        data = 97 + torch.floor(26 * data).int()
        offset = 0
        print(num_tokens, dim, file=f)

        token = 'test'
        embed = list(map(str, torch.randn(dim).tolist()))
        print(f'{token} {" ".join(embed)}', file=f)

        for i in range(num_tokens - 1):
            token_len = random.randint(1, 10)
            token = ''.join(map(chr, data[offset: offset + token_len]))
            offset += token_len
            embed = list(map(str, torch.randn(dim).tolist()))
            print(f'{token} {" ".join(embed)}', file=f)


class TestStaticEmbedding(unittest.TestCase):
    def test_static_embedding(self):
        with tempfile.TemporaryDirectory(f'test_embed') as data_dir:
            embedding_file = Path(data_dir) / 'embedding.txt'

            num_tokens = random.randint(30, 50)
            create_dummy_data(embedding_file, num_tokens=num_tokens)
            module = StaticEmbedding(str(embedding_file))

            out_file = Path(data_dir) / 'out_file'
            with open(out_file, 'wb') as f:
                pickle.dump(module, f)

            vec1, status = module.processor.query('test')

            with open(out_file, 'rb') as f:
                m = pickle.load(f)

            vec2, status = m.processor.query('test')

            self.assertTrue('test' in module.processor)
            self.assertTrue('test' in m.processor)
            self.assertEqual(num_tokens, module.length)
            self.assertEqual(num_tokens, m.length)
            numpy.testing.assert_equal(vec1, vec2)
