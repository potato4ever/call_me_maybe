import unittest

import numpy as np

from src.decoding import (
    ConstraintCache,
    TokenCandidateIndex,
    TokenTrie,
    _best_allowed_token,
)
from src.grammar import TrieGrammar


class FakeModel:
    def __init__(self) -> None:
        self.tokens = {
            "fn": 0,
            "_add": 1,
            "_numbers": 2,
            "fn_greet": 3,
            "true": 4,
            "false": 5,
            "hello": 6,
            '"': 7,
            "4": 8,
            "2": 9,
        }

    def encode(self, text: str):
        class FakeTensor:
            def __init__(self, ids):
                self.ids = ids

            def tolist(self):
                return [self.ids]

        pieces = {
            "fn_add_numbers": [0, 1, 2],
            "fn_greet": [3],
            "true": [4],
            "false": [5],
        }

        return FakeTensor(pieces[text])

    def decode_token(self, token_id: int) -> str:
        reverse = {
            value: key
            for key, value in self.tokens.items()
        }

        return reverse[token_id]

    def get_vocab(self):
        return self.tokens


class TestTokenTrie(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()

    def test_complete_values_use_actual_token_sequences(self) -> None:
        trie = TokenTrie(
            ["fn_add_numbers", "fn_greet"],
            self.model,
        )

        allowed, complete = trie.allowed([])

        self.assertEqual(
            allowed.tolist(),
            [0, 3],
        )
        self.assertFalse(complete)

        allowed, complete = trie.allowed([0])

        self.assertEqual(
            allowed.tolist(),
            [1],
        )
        self.assertFalse(complete)

        allowed, complete = trie.allowed([0, 1])

        self.assertEqual(
            allowed.tolist(),
            [2],
        )
        self.assertFalse(complete)

        allowed, complete = trie.allowed([0, 1, 2])

        self.assertEqual(
            allowed.tolist(),
            [],
        )
        self.assertTrue(complete)


class TestGreedySelection(unittest.TestCase):
    def test_only_allowed_ids_are_considered(self) -> None:
        logits = np.asarray(
            [1.0, 10.0, 3.0, 2.0],
            dtype=np.float32,
        )

        allowed = np.asarray(
            [0, 2, 3],
            dtype=np.int64,
        )

        self.assertEqual(
            _best_allowed_token(logits, allowed),
            2,
        )


class TestConstraintCache(unittest.TestCase):
    def test_same_closed_values_are_cached(self) -> None:
        model = FakeModel()
        cache = ConstraintCache(model)

        first = cache.trie(["true", "false"])
        second = cache.trie(["true", "false"])

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
