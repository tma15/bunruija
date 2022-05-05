import unittest

from bunruija.tokenizers.mecab_tokenizer import MeCabTokenizer
from bunruija.tokenizers.space_tokenizer import SpaceTokenizer


class TestSpaceTokenizer(unittest.TestCase):
    def test_tokenize(self):
        self.tokenizer = SpaceTokenizer()
        text = "This is a test sentence."
        tokens = self.tokenizer(text)
        expected = ["This", "is", "a", "test", "sentence."]
        self.assertEqual(expected, tokens)

    def test_tokenize_redundant_spaces(self):
        self.tokenizer = SpaceTokenizer()
        text = "This is a test   sentence."
        tokens = self.tokenizer(text)
        expected = ["This", "is", "a", "test", "sentence."]
        self.assertEqual(expected, tokens)


class TestMeCabTokenizer(unittest.TestCase):
    def test_tokenize(self):
        self.tokenizer = MeCabTokenizer()
        text = "昨日はご飯を食べた"
        tokens = self.tokenizer(text)
        expected = ["昨日", "は", "ご飯", "を", "食べ", "た"]
        self.assertEqual(expected, tokens)

    def test_tokenize_lemmatize(self):
        self.tokenizer = MeCabTokenizer(lemmatize=True)
        text = "昨日はご飯を食べた"
        tokens = self.tokenizer(text)
        expected = ["昨日", "は", "御飯", "を", "食べる", "た"]
        self.assertEqual(expected, tokens)

    def test_tokenize_pos_filter(self):
        self.tokenizer = MeCabTokenizer(exclude_pos=["助詞", "助動詞"])
        text = "昨日はご飯を食べた"
        tokens = self.tokenizer(text)
        expected = ["昨日", "ご飯", "食べ"]
        self.assertEqual(expected, tokens)
