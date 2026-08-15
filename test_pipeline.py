# ABOUTME: End-to-end test of decoding.py + pipeline.py using a small hand-written
# ABOUTME: fake vocabulary and a scripted fake model, so the full constrained decoding
# ABOUTME: loop is exercised without needing torch, network access, or real weights.

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from src.grammar import NumberGrammar, StringGrammar, TrieGrammar
from src.decoding import constrained_generate
from src.pipeline import process_prompt
from src.vocab import VocabIndex, _bytes_to_unicode


def _make_vocab_file(tokens: List[str]) -> str:
    byte_encoder = _bytes_to_unicode()

    def encode_text(text: str) -> str:
        return "".join(byte_encoder[b] for b in text.encode("utf-8"))

    raw_vocab = {encode_text(tok): idx for idx, tok in enumerate(tokens)}
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "vocab.json"
    path.write_text(json.dumps(raw_vocab), encoding="utf-8")
    return str(path)


class FakeTensor:
    """Minimal stand-in for the torch.Tensor returned by Small_LLM_Model.encode()."""

    def __init__(self, ids: List[int]):
        self._ids = ids

    def tolist(self) -> List[List[int]]:
        return [self._ids]


class ScriptedFakeModel:
    """A deterministic fake model that scores a scripted sequence of tokens highest.

    Every call to get_logits_from_input_ids advances an internal counter and gives a
    large logit boost to whichever vocab token exactly matches ``script[counter]``
    (clamped to the last entry once the script is exhausted). This lets tests fully
    control what the "model" prefers to generate at each step, independent of the
    (irrelevant, in tests) actual token ids.
    """

    def __init__(self, vocab_tokens: List[str], script: List[str]) -> None:
        self._vocab_tokens = vocab_tokens
        self._script = script
        self._step = 0

    def encode(self, text: str) -> FakeTensor:
        return FakeTensor([0])  # content is irrelevant to this fake

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        target = self._script[min(self._step, len(self._script) - 1)]
        self._step += 1
        return [10.0 if tok == target else 0.0 for tok in self._vocab_tokens]


class TestConstrainedGenerateWithFakeModel(unittest.TestCase):
    def test_generates_expected_string(self) -> None:
        tokens = ["hello", '"', "world", "x"]
        vocab_path = _make_vocab_file(tokens)
        vocab_index = VocabIndex(vocab_path)
        model = ScriptedFakeModel(tokens, script=["hello", '"'])

        result = constrained_generate(model, vocab_index, "", StringGrammar(), max_tokens=10)
        self.assertEqual(result, 'hello"')

    def test_generates_expected_number(self) -> None:
        tokens = ["4", "2", "0", "."]
        vocab_path = _make_vocab_file(tokens)
        vocab_index = VocabIndex(vocab_path)
        # Model "wants" to keep emitting "4" forever; the grammar has no explicit
        # closing symbol for numbers, so generation stops once the safety cap is hit.
        model = ScriptedFakeModel(tokens, script=["4"])

        result = constrained_generate(model, vocab_index, "", NumberGrammar(), max_tokens=3)
        self.assertTrue(result.startswith("4"))
        self.assertTrue(all(c == "4" for c in result))

    def test_generates_expected_enum_value(self) -> None:
        tokens = ["fn_greet", "fn_add_numbers", "x"]
        vocab_path = _make_vocab_file(tokens)
        vocab_index = VocabIndex(vocab_path)
        model = ScriptedFakeModel(tokens, script=["fn_greet"])

        grammar = TrieGrammar(["fn_greet", "fn_add_numbers"])
        result = constrained_generate(model, vocab_index, "", grammar, max_tokens=10)
        self.assertEqual(result, "fn_greet")


class TestPipelineWithFakeModel(unittest.TestCase):
    def test_process_prompt_end_to_end(self) -> None:
        from src.models import FunctionDefinition, ParameterType

        functions = [
            FunctionDefinition(
                name="fn_greet",
                description="Greet someone",
                parameters={"name": ParameterType(type="string")},
                returns=ParameterType(type="string"),
            ),
            FunctionDefinition(
                name="fn_add_numbers",
                description="Add numbers",
                parameters={
                    "a": ParameterType(type="number"),
                    "b": ParameterType(type="number"),
                },
                returns=ParameterType(type="number"),
            ),
        ]
        functions_by_name = {fn.name: fn for fn in functions}

        tokens = ["fn_greet", "fn_add_numbers", '"', "shrek", "x"]
        vocab_path = _make_vocab_file(tokens)
        vocab_index = VocabIndex(vocab_path)

        # Step-by-step script: pick fn_greet, then generate "shrek" then close the
        # string with a quote token.
        model = ScriptedFakeModel(tokens, script=["fn_greet", "shrek", '"'])

        result = process_prompt(model, vocab_index, functions_by_name, functions, "Greet shrek")

        self.assertEqual(result.name, "fn_greet")
        self.assertEqual(result.parameters, {"name": "shrek"})
        self.assertEqual(result.prompt, "Greet shrek")


if __name__ == "__main__":
    unittest.main()
