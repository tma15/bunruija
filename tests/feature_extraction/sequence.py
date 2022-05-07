import unittest

from bunruija.feature_extraction.sequence import SequenceVectorizer


class TestSequenceVectorizer(unittest.TestCase):
    def test_fit_transform(self):
        vectorizer = SequenceVectorizer()
        texts = ["今日はいい天気だ", "ごはんにはラーメンを食べた"]
        csr_matrix, raw_tokens = vectorizer.fit_transform(texts)
        dense_matrix = csr_matrix.toarray()

        ids0 = dense_matrix[0]
        tokens0 = [vectorizer.dictionary.get_element(idx) for idx in ids0]
        expected0 = ["今日", "は", "いい", "天気", "だ", "<pad>", "<pad>"]
        self.assertEqual(expected0, tokens0)

        start = csr_matrix.indptr[0]
        end = csr_matrix.indptr[1]
        tokens01 = raw_tokens[start:end]
        expected01 = ["今日", "は", "いい", "天気", "だ"]
        self.assertEqual(tokens01, expected01)

        ids1 = dense_matrix[1]
        tokens1 = [vectorizer.dictionary.get_element(idx) for idx in ids1]
        expected1 = ["ごはん", "に", "は", "ラーメン", "を", "食べ", "た"]
        self.assertEqual(expected1, tokens1)
