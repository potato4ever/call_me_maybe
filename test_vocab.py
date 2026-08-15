# ABOUTME: Unit tests for VocabIndex's byte-level BPE decoding, using a small
# ABOUTME: hand-written vocab.json fixture instead of a real (huge) model vocab.

import json
import tempfile
import unittest
from pathlib import Path

from src.vocab import VocabIndex, _bytes_to_unicode


class TestBytesToUnicode(unittest.TestCase):
    def test_is_a_bijection_over_256_bytes(self) -> None:
        mapping = _bytes_to_unicode()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(len(set(mapping.values())), 256)


class TestVocabIndex(unittest.TestCase):
    def setUp(self) -> None:
        byte_encoder = _bytes_to_unicode()

        def encode_text(text: str) -> str:
            return "".join(byte_encoder[b] for b in text.encode("utf-8"))

        # "Ġ" (byte_encoder[0x20]) is the conventional leading-space marker.
        space = byte_encoder[0x20]
        raw_vocab = {
            encode_text("hello"): 0,
            f"{space}{encode_text('world')}": 1,
            encode_text('"'): 2,
            encode_text(","): 3,
        }

        tmp_dir = tempfile.mkdtemp()
        self.vocab_path = Path(tmp_dir) / "vocab.json"
        self.vocab_path.write_text(json.dumps(raw_vocab), encoding="utf-8")

    def test_decodes_plain_token(self) -> None:
        index = VocabIndex(str(self.vocab_path))
        self.assertEqual(index.decode_token(0), "hello")

    def test_decodes_leading_space_token(self) -> None:
        index = VocabIndex(str(self.vocab_path))
        self.assertEqual(index.decode_token(1), " world")

    def test_decodes_punctuation(self) -> None:
        index = VocabIndex(str(self.vocab_path))
        self.assertEqual(index.decode_token(2), '"')
        self.assertEqual(index.decode_token(3), ",")

    def test_items_and_len(self) -> None:
        index = VocabIndex(str(self.vocab_path))
        self.assertEqual(len(index), 4)
        self.assertEqual(dict(index.items())[0], "hello")

    def test_unknown_id_returns_empty_string(self) -> None:
        index = VocabIndex(str(self.vocab_path))
        self.assertEqual(index.decode_token(999), "")


if __name__ == "__main__":
    unittest.main()
