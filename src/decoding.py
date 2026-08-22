try:
    from typing import Any, Dict, List, Protocol, Tuple

    import numpy as np

    from .grammar import Grammar, TrieGrammar

    from pathlib import Path
    import json
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
        file=sys.stderr,
    )


class DecodingError(Exception):
    """Raised when constrained decoding cannot make progress."""


class LLMModel(Protocol):
    """The model interface required by the constrained decoder."""

    def encode(self, text: str) -> Any:
        ...

    def get_path_to_vocab_file(self) -> str:
        ...

    def decode(self, token_id: int) -> str:
        ...

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        ...

def get_vocab(model: LLMModel) -> Dict[str, int]: 
    path = Path(model.get_path_to_vocab_file())
    if not path.is_file():
        raise DecodingError("no vocab file founded")
    try:
        vocab = path.read_text()
    except OSError as exc:
        raise DecodingError(f"could not read {path}: {exc}")
    try:
        v = json.loads(vocab)
    except json.JSONDecodeError as exc:
        raise DecodingError(
            f"invalid JSON in vocab file {path}: {exc}"
        )
    if not isinstance(v, dict):
        raise DecodingError(f"the vocab content should be dictionary")
    return v

def _encode_to_ids(model: LLMModel, text: str) -> List[int]:
    """Encode text and normalize the result to a plain list of token IDs."""
    encoded = model.encode(text)

    to_list = getattr(encoded, "tolist", None)

    if callable(to_list):
        ids = to_list()
    else:
        ids = encoded

    # encode() normally returns [batch, sequence].
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    if not ids:
        raise DecodingError(
            "model.encode() returned an empty token sequence"
        )

    return [int(value) for value in ids]


class TokenTrieNode:
    """A node in a token-ID trie."""

    def __init__(self) -> None:
        self.children: Dict[int, "TokenTrieNode"] = {}
        self.complete = False


class TokenTrie:
    """Trie of complete tokenizer-produced sequences.

    This is used only for closed grammars such as function names and
    boolean values, where we already know every possible valid value.
    """

    def __init__(
        self,
        values: List[str],
        model: LLMModel,
    ) -> None:
        self.root = TokenTrieNode()

        for value in values:
            ids = _encode_to_ids(model, value)
            ids.extend(_encode_to_ids(model, "\""))

            node = self.root

            for token_id in ids:
                node = node.children.setdefault(
                    token_id,
                    TokenTrieNode(),
                )
            node.complete = True
            node = self.root

    def allowed(
        self,
        generated_ids: List[int],
    ) -> Tuple[np.ndarray, bool]:
        """Return valid next token IDs and whether the current value is complete."""
        node = self.root

        for token_id in generated_ids:
            node = node.children.get(token_id)

            if node is None:
                return np.empty(0, dtype=np.int64)

        allowed_ids = np.array(
            list(node.children.keys()),
            dtype=np.int64,
        )

        return allowed_ids


class ConstraintCache:
    """Cache token tries for closed grammars."""

    def __init__(self, model: LLMModel) -> None:
        self._model = model
        self._tries: Dict[Tuple[str, ...], TokenTrie] = {}

    def trie(self, options: Tuple[str]) -> TokenTrie:
        key = tuple(options)

        cached = self._tries.get(key)

        if cached is not None:
            return cached

        trie = TokenTrie(options, self._model)
        self._tries[key] = trie

        return trie


def _is_closed_grammar(grammar: Grammar) -> bool:
    """Return whether the grammar represents a fixed set of values."""
    return isinstance(grammar, TrieGrammar)


def _best_allowed_token(
    logits: np.ndarray,
    allowed_ids: np.ndarray,
) -> int | None:
    """Return the highest-logit token among the allowed token IDs."""
    if allowed_ids.size == 0:
        return None

    valid_ids = allowed_ids[
        (allowed_ids >= 0) & (allowed_ids < logits.size)
    ]

    if valid_ids.size == 0:
        return None

    values = logits[valid_ids]
    best_index = int(np.argmax(values))

    return int(valid_ids[best_index])


def constrained_generate(
    model: LLMModel,
    context_text: str,
    grammar: Grammar,
    constraint_cache: ConstraintCache,
    max_tokens: int = 60,
) -> str:
    """Generate a grammar-constrained continuation.

    For closed grammars, a token-ID trie dramatically reduces the candidate
    set.

    For dynamic grammars such as numbers and strings, every vocabulary token
    is checked directly against the grammar. This is intentionally simple:
    correctness is more important than optimization at this stage.
    """

    generated = ""
    generated_ids: List[int] = []

    trie = None

    if _is_closed_grammar(grammar):
        trie = constraint_cache.trie(grammar._options)

    # Get the vocabulary once.
    vocab = get_vocab(model)

    vocabulary_ids = np.asarray(
        list(vocab.values()),
        dtype=np.int64,
    )

    for step in range(max_tokens):
        input_ids = _encode_to_ids(
            model,
            context_text + generated,
        )

        try:
            raw_logits = model.get_logits_from_input_ids(
                input_ids
            )
        except Exception as exc:
            raise DecodingError(
                f"model inference failed: {exc}"
            ) from exc

        logits = np.asarray(
            raw_logits,
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # Closed grammar
        # ---------------------------------------------------------------
        if trie is not None:
            allowed_ids = trie.allowed(
                generated_ids
            )




            best_id = _best_allowed_token(
                logits,
                allowed_ids,
            )

            if best_id is None:
                if grammar.is_complete(generated):
                    break
                raise DecodingError("no valid token for generated")

            best_text = model.decode(best_id)
            best_logit = float(logits[best_id])

            if not best_text:
                raise DecodingError(f"chosen token {best_id} is empty text")


                # A token with no decoded text cannot help us generate a value.
                generated_ids.append(best_id)
                continue

            status = grammar.check(
                generated,
                best_text,
            )

            if status == "invalid":
                raise DecodingError(f"unknow function {best_text}")
            if best_text != "\"":
                generated += best_text
                generated_ids.append(best_id)

            if status == "complete":
                break

            continue

        # ---------------------------------------------------------------
        # Dynamic grammar
        # ---------------------------------------------------------------
        #
        # Every vocabulary token is checked directly against the grammar.
        # We intentionally do not infer validity from token characters alone.
        # ---------------------------------------------------------------

        best_id = None
        best_text = ""
        best_status = "invalid"
        best_logit = float("-inf")


        for token_id in vocabulary_ids.tolist():
            token_text = model.decode(token_id)

            if not token_text:
                continue

            status = grammar.check(
                generated,
                token_text,
            )

            logit = float(logits[token_id])

            if status == "invalid":
                continue


            if best_id is None or logit > best_logit:
                best_id = token_id
                best_text = token_text
                best_status = status
                best_logit = logit

        # No valid token was found.
        if best_id is None:
            raise DecodingError(
                "no valid token available to continue generation: "
                f"{generated!r}"
            )

        if best_status == "complete":
            completion_text = grammar.consume_completion(
                generated,
                best_text,
            )

            generated += completion_text
            generated_ids.append(best_id)

            break

        generated += best_text
        generated_ids.append(best_id)
    return generated
if __name__ == "__main__":
    from llm_sdk.llm_sdk import Small_LLM_Model
    model = Small_LLM_Model()
    t = tuple(["potato", "tomtato", "1234"])
    print(t)
    TokenTrie(t, model)


