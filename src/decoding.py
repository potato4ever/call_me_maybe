# ABOUTME: Constrained decoding engine.
# ABOUTME: At every generation step, evaluate the vocabulary tokens against
# ABOUTME: the active grammar and greedily select the highest-logit valid token.

from typing import Any, Dict, List, Protocol

import numpy as np

from .grammar import Grammar, TrieGrammar


class DecodingError(Exception):
    """Raised when constrained decoding cannot make progress."""


class LLMModel(Protocol):
    """The model interface required by the constrained decoder."""

    def encode(self, text: str) -> Any:
        ...

    def decode_token(self, token_id: int) -> str:
        ...

    def get_vocab(self) -> Dict[str, int]:
        ...

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        ...


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

    __slots__ = ("children", "complete")

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

            node = self.root

            for token_id in ids:
                node = node.children.setdefault(
                    token_id,
                    TokenTrieNode(),
                )

            node.complete = True

    def allowed(
        self,
        generated_ids: List[int],
    ) -> tuple[np.ndarray, bool]:
        """Return valid next token IDs and whether the current value is complete."""
        node = self.root

        for token_id in generated_ids:
            node = node.children.get(token_id)

            if node is None:
                return np.empty(0, dtype=np.int64), False

        allowed_ids = np.fromiter(
            node.children.keys(),
            dtype=np.int64,
        )

        return allowed_ids, node.complete


class ConstraintCache:
    """Cache token tries for closed grammars."""

    def __init__(self, model: LLMModel) -> None:
        self._model = model
        self._tries: Dict[tuple[str, ...], TokenTrie] = {}

    def trie(self, options: List[str]) -> TokenTrie:
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

def _print_dynamic_candidates(
    candidates: List[tuple[float, int, str, str]],
    limit: int = 10,
) -> None:
    """Print only the highest-logit valid candidates."""

    for logit, token_id, token_text, status in candidates[:limit]:
        print(
            f"    {logit:7.2f}  "
            f"{token_text!s:12} -> {status}"
        )

def constrained_generate(
    model: LLMModel,
    context_text: str,
    grammar: Grammar,
    *,
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
        assert isinstance(grammar, TrieGrammar)

        trie = constraint_cache.trie(grammar._options)

    # Get the vocabulary once.
    vocab = model.get_vocab()

    vocabulary_ids = np.asarray(
        list(vocab.values()),
        dtype=np.int64,
    )

    for step in range(max_tokens):
        print("\n" + "=" * 80)
        print("ACTUAL MODEL INPUT")
        print("=" * 80)
        print(context_text + generated)
        print("=" * 80)
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
        print(f"\nSTEP {step} | {type(grammar).__name__}")
        print(f"generated: {generated!r}")

        # ---------------------------------------------------------------
        # Closed grammar
        # ---------------------------------------------------------------
        if trie is not None:
            allowed_ids, trie_complete = trie.allowed(
                generated_ids
            )

            print("\n" + "-" * 80)
            print("CLOSED GRAMMAR STATE")
            print("-" * 80)
            print("Grammar:", type(grammar).__name__)
            print("Generated IDs:", generated_ids)
            print("Allowed IDs:", allowed_ids.tolist())
            print("Trie complete:", trie_complete)
            print("-" * 80)

            if trie_complete:
                print("TRIE SAYS VALUE IS COMPLETE")
                break


            best_id = _best_allowed_token(
                logits,
                allowed_ids,
            )

            if best_id is None:
                print("NO VALID TOKEN SELECTED")

                if grammar.is_complete(generated):
                    print("Grammar itself considers the value complete.")
                    break

                suffix = grammar.force_close(generated)

                print(
                    "Grammar force_close returned:",
                    repr(suffix),
                )

                if not suffix:
                    raise DecodingError(
                        "grammar stalled before completing value: "
                        f"{generated!r}"
                    )

                generated += suffix
                break

            best_text = model.decode_token(best_id)
            best_logit = float(logits[best_id])

            if not best_text:
                print(
                    "CHOSEN TOKEN HAS EMPTY TEXT:",
                    f"id={best_id}",
                    f"logit={best_logit:.4f}",
                )

                # A token with no decoded text cannot help us generate a value.
                generated_ids.append(best_id)
                continue

            status = grammar.check(
                generated,
                best_text,
            )

            print("\n" + "-" * 80)
            print("CHOSEN TOKEN")
            print("-" * 80)
            print(f"id:       {best_id}")
            print(f"text:     {best_text!r}")
            print(f"logit:    {best_logit:.4f}")
            print(f"status:   {status}")
            print(f"previous: {generated!r}")
            print(f"new:      {(generated + best_text)!r}")
            print("-" * 80)

            if status == "invalid":
                print(
                    "WARNING: trie-selected token was rejected by grammar."
                )

                valid_ids = [
                    token_id
                    for token_id in allowed_ids.tolist()
                    if 0 <= int(token_id) < logits.size
                    and grammar.check(
                        generated,
                        model.decode_token(int(token_id)),
                    ) != "invalid"
                ]

                print(
                    "Grammar-valid subset of trie candidates:",
                    valid_ids,
                )

                best_id = _best_allowed_token(
                    logits,
                    np.asarray(
                        valid_ids,
                        dtype=np.int64,
                    ),
                )

                if best_id is None:
                    if grammar.is_complete(generated):
                        break

                    suffix = grammar.force_close(generated)

                    print(
                        "Grammar force_close returned:",
                        repr(suffix),
                    )

                    if not suffix:
                        raise DecodingError(
                            "grammar stalled before completing value: "
                            f"{generated!r}"
                        )

                    generated += suffix
                    break

                best_text = model.decode_token(best_id)
                best_logit = float(logits[best_id])

                status = grammar.check(
                    generated,
                    best_text,
                )

                print(
                    "FALLBACK CHOSEN TOKEN:",
                    f"id={best_id}",
                    f"text={best_text!r}",
                    f"logit={best_logit:.4f}",
                    f"status={status}",
                )

            generated += best_text
            generated_ids.append(best_id)

            print(
                "GENERATED NOW:",
                repr(generated),
            )

            if status == "complete":
                print("GRAMMAR COMPLETE")
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

        debug_candidates: List[
            tuple[float, int, str, str]
        ] = []

        invalid_count = 0
        empty_count = 0

        for token_id in vocabulary_ids.tolist():
            token_id = int(token_id)

            if token_id < 0 or token_id >= logits.size:
                continue

            token_text = model.decode_token(token_id)

            if not token_text:
                empty_count += 1
                continue

            status = grammar.check(
                generated,
                token_text,
            )

            logit = float(logits[token_id])

            if status == "invalid":
                invalid_count += 1
                continue

            debug_candidates.append(
                (
                    logit,
                    token_id,
                    repr(token_text),
                    status,
                )
            )

            if best_id is None or logit > best_logit:
                best_id = token_id
                best_text = token_text
                best_status = status
                best_logit = logit

        debug_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _print_dynamic_candidates(
            debug_candidates,
        )

        # No valid token was found.
        if best_id is None:
            print("NO VALID TOKEN FOUND")

            if grammar.is_complete(generated):
                print("Grammar itself considers the value complete.")
                break

            suffix = grammar.force_close(generated)

            print(
                "Grammar force_close returned:",
                repr(suffix),
            )

            if not suffix:
                raise DecodingError(
                    "grammar stalled before completing value: "
                    f"{generated!r}"
                )

            generated += suffix
            break
        print(
            f"selected: {best_text!r} -> {best_status}"
        )

        if best_status == "complete":
            completion_text = grammar.consume_completion(
                generated,
                best_text,
            )

            generated += completion_text
            generated_ids.append(best_id)

            print(
                "COMPLETION TOKEN:",
                repr(best_text),
            )
            print(
                "SEMANTIC TEXT:",
                repr(completion_text),
            )
            print("GRAMMAR COMPLETE")

            break

        generated += best_text
        generated_ids.append(best_id)

    else:
        print("\n" + "=" * 80)
        print("MAXIMUM GENERATION LENGTH REACHED")
        print("=" * 80)
        print("Generated:", repr(generated))

        if not grammar.is_complete(generated):
            suffix = grammar.force_close(generated)

            print(
                "Grammar force_close returned:",
                repr(suffix),
            )

            if not suffix:
                raise DecodingError(
                    "generation limit reached before completion: "
                    f"{generated!r}"
                )

            generated += suffix

    print("\n" + "=" * 80)
    print("FINAL CONSTRAINED GENERATION")
    print("=" * 80)
    print(repr(generated))
    print("Token IDs:", generated_ids)
    print("=" * 80)

    return generated
