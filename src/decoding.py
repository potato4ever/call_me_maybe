"""Constrained decoding implementation for LLM function calling.

This module provides the core logic for intercepting LLM logits and
restricting token selection based on a grammar or a trie of valid values.
"""
import sys
try:
    from typing import (Any, Dict, List, Protocol, Tuple, Optional,
                        Sequence, cast)
    import numpy as np
    from .grammar import Grammar, TrieGrammar
    from pathlib import Path
    import json
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}"
    )
    sys.exit(1)


class DecodingError(Exception):
    """Raised when constrained decoding cannot make progress."""


class LLMModel(Protocol):
    """The model interface required by the constrained decoder.

    This protocol ensures that any model passed to the decoder has the
    required methods for tokenization, inference, and vocabulary access.
    """

    def encode(self, text: str) -> Any:
        """Encode text into token IDs.

        Args:
            text: The text to tokenize.

        Returns:
            Any: The encoded token IDs returned by the model tokenizer.
        """
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the model vocabulary file.

        Returns:
            str: The filesystem path to the vocabulary JSON file.
        """
        ...

    def decode(self, ids: Any) -> str:
        """Decode token IDs into text.

        Args:
            ids: The token ID or token IDs to decode.

        Returns:
            str: The decoded text.
        """
        ...

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Compute logits for the next token.

        Args:
            input_ids: The token IDs representing the current input.

        Returns:
            List[float]: The logits for the next-token vocabulary.
        """
        ...


def get_vocab(model: LLMModel) -> Dict[str, int]:
    """Retrieve and validate the model's vocabulary.

    Args:
        model: The LLM model instance.

    Returns:
        Dict[str, int]: The mapping of tokens to IDs.

    Raises:
        DecodingError: If the vocab file is missing, unreadable, or invalid.
    """
    path = Path(model.get_path_to_vocab_file())
    if not path.is_file():
        raise DecodingError("no vocab file founded")
    try:
        vocab = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DecodingError(f"could not read {path}: {exc}")
    try:
        v = json.loads(vocab)
    except json.JSONDecodeError as exc:
        raise DecodingError(
            f"invalid JSON in vocab file {path}: {exc}"
        )
    if not isinstance(v, dict):
        raise DecodingError("the vocab content should be dictionary")
    return v


def _encode_to_ids(model: LLMModel, text: str) -> List[int]:
    """Encode text and normalize the result to a plain list of token IDs.

    Args:
        model: The LLM model instance.
        text: The string to encode.

    Returns:
        List[int]: A flat list of token IDs.

    Raises:
        DecodingError: If the model returns an empty sequence.
    """
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
    """A node in a token-ID trie.

    Attributes:
        children: Map of token ID to the next node.
        complete: Boolean indicating if this node represents a valid endpoint.
    """

    def __init__(self) -> None:
        """Initialize a trie node."""
        self.children: Dict[int, "TokenTrieNode"] = {}
        self.complete: bool = False


class TokenTrie:
    """Trie of complete tokenizer-produced sequences.

    Used for closed grammars (like function names or booleans) where all
    possible valid string values are known in advance.
    """

    def __init__(
        self,
        values: Sequence[str],
        model: LLMModel,
    ) -> None:
        """Build a token trie from a list of strings.

        Args:
            values: Sequence of valid string options.
            model: The LLM model used for encoding.
        """
        self.root = TokenTrieNode()

        for value in values:
            ids = _encode_to_ids(model, value)
            ids.extend(_encode_to_ids(model, "\""))

            node: TokenTrieNode = self.root

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
    ) -> np.ndarray:

        """Return valid next token IDs based on the sequence generated so far.

        Args:
            generated_ids: Token IDs generated in the current phase.

        Returns:
            np.ndarray: An array of allowed token IDs.
        """
        node = self.root  # no Optional annotation needed

        for token_id in generated_ids:
            next_node = node.children.get(token_id)
            if next_node is None:
                return np.empty(0, dtype=np.int64)
            node = next_node

        return np.array(list(node.children.keys()), dtype=np.int64)


class ConstraintCache:
    """Cache for token tries to avoid redundant re-encoding.

    Attributes:
        model: The LLM model used for encoding values.
    """

    def __init__(self, model: LLMModel) -> None:
        """Initialize the cache."""
        self._model = model
        self._tries: Dict[Tuple[str, ...], TokenTrie] = {}

    def trie(self, options: Tuple[str, ...]) -> TokenTrie:
        """Retrieve a cached trie or create a new one.

        Args:
            options: Tuple of valid strings for the grammar.

        Returns:
            TokenTrie: The trie for the provided options.
        """
        key = tuple(options)

        cached = self._tries.get(key)

        if cached is not None:
            return cached

        trie = TokenTrie(options, self._model)
        self._tries[key] = trie

        return trie


def _is_closed_grammar(grammar: Grammar) -> bool:
    """Check if the grammar represents a fixed set of values."""
    return isinstance(grammar, TrieGrammar)


def _best_allowed_token(
    logits: np.ndarray,
    allowed_ids: np.ndarray,
) -> Optional[int]:
    """Find the highest-logit token within the allowed set.

    Args:
        logits: Full probability distribution from the LLM.
        allowed_ids: Subset of token IDs permitted by the grammar.

    Returns:
        Optional[int]: The best token ID, or None if no valid tokens exist.
    """
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
    """Generate text while strictly enforcing grammar constraints.

    Args:
        model: The LLM model instance.
        context_text: The initial prompt text.
        grammar: The grammar object defining valid transitions.
        constraint_cache: Cache for reusable token tries.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        str: The generated, valid string.

    Raises:
        DecodingError: If no valid tokens are found or inference fails.
    """

    generated = ""
    generated_ids: List[int] = []

    trie: Optional[TokenTrie] = None

    if _is_closed_grammar(grammar):
        trie = constraint_cache.trie(cast(TrieGrammar, grammar)._options)

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
