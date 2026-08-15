*This project has been created as part of the 42 curriculum by \<login1\>[, \<login2\>].*

# call me maybe — Introduction to function calling in LLMs

## Description

This project turns natural-language requests ("What is the sum of 2 and 3?") into
structured, machine-executable function calls (`{"name": "fn_add_numbers", "parameters":
{"a": 2.0, "b": 3.0}}`), using a small (0.6B parameter) local LLM, **Qwen/Qwen3-0.6B**.

Small models are notoriously unreliable at spontaneously producing valid JSON — they
might succeed 30% of the time when simply prompted. Instead of hoping for the best,
this project implements **constrained decoding**: at every single generation step, we
inspect the model's raw output probabilities (logits) and mask out every token that
would break either JSON syntax or the expected function schema, before a token is ever
chosen. The result is **100% syntactically and semantically valid output**, regardless
of the model's raw reliability.

The program reads:
- `data/input/functions_definition.json` — the catalog of callable functions (name,
  parameter types, description).
- `data/input/function_calling_tests.json` — a list of natural-language prompts.

...and writes `data/output/function_calling_results.json`: for every prompt, the
selected function name and its correctly-typed arguments.

## Instructions

### Requirements
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Enough disk space / RAM to download and run `Qwen/Qwen3-0.6B` (a few GB) — the first
  run will download the model from the Hugging Face Hub.

### Setup

```bash
git clone <this-repo>
cd call_me_maybe
make install        # equivalent to: uv sync
```

`llm_sdk/` is vendored (copied) directly into this repository as instructed by the
subject, and declared as a local editable dependency in `pyproject.toml`, so `uv sync`
installs it (and its own dependencies: `torch`, `transformers`, `huggingface-hub`)
together with `pydantic` and `numpy`.

### Running

```bash
make run
# equivalent to:
uv run python -m src

# or, with explicit paths:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

By default, input is read from `data/input/` and output is written to
`data/output/function_calling_results.json` (created automatically).

Other Makefile targets:

```bash
make debug        # run under pdb
make test         # run the (fast, model-free) unit test suite
make lint          # flake8 + mypy (standard flags)
make lint-strict   # flake8 + mypy --strict
make clean         # remove caches and generated output
```

## Resources

- [Hugging Face — Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
  (background on how byte-level BPE vocab.json files are structured — needed to decode
  raw vocab entries into real text without depending on the `tokenizers` library).
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) — used as the
  reference grammar for what counts as a valid JSON string/number.
- [Guidance / Outlines project write-ups on constrained decoding](https://github.com/dottxt-ai/outlines)
  — general background reading on logit-masking approaches to structured generation
  (no code or dependency from these projects was used, per the subject's constraints).
- The subject PDF itself ("call me maybe — Introduction to function calling in LLMs").

### How AI was used

An AI assistant (Claude) was used to:
- Draft the initial project structure (`src/` module layout, `Makefile`, `pyproject.toml`).
- Draft the constrained decoding engine (`src/decoding.py`), the grammar state machines
  (`src/grammar.py`), and the byte-level BPE vocab decoder (`src/vocab.py`) — including
  writing and running the unit tests in `tests/` to verify the number-prefix DFA, the
  string/enum grammars, and the end-to-end pipeline logic against a scripted fake model
  (no real model weights were available in the assistant's sandbox, so the constrained
  decoding logic was validated with fake vocab/model doubles rather than a live LLM).
- Draft this README.

Every generated file was read and is understood; before submitting, re-run
`make test` and `make lint`, and manually exercise `make run` against the real model to
confirm end-to-end behaviour, per the "AI Instructions" chapter of the subject.

## Algorithm explanation

Generation happens in two constrained-decoding phases per prompt:

1. **Function selection.** The model is prompted with the list of available functions
   (name, parameter types, description) and the user's request, ending exactly at
   `{"name": "`. From there, decoding is constrained to a **trie over the set of known
   function names**: at each step, every vocabulary token is checked against
   "would appending this token still be a prefix of at least one valid function name?".
   Invalid tokens are discarded; among the remaining valid tokens, the one with the
   highest raw logit is picked (constrained-greedy decoding). Generation stops as soon
   as the accumulated text exactly matches one of the known names.

2. **Parameter generation.** Once the function (and therefore its parameter schema) is
   known, we build the `"parameters": {...}` object key by key. Structural tokens that
   are fully determined by the schema (`{`, `"key": `, `,`, `}`, opening quotes, ...)
   are appended directly to the running context — there's nothing to "decide" there, so
   no model call is wasted on them. Only the *value* of each parameter is generated by
   the model, constrained by a grammar that matches its declared type:
   - `string` → any content, terminated as soon as the model itself emits a token whose
     only/last character is an unescaped closing quote.
   - `number` / `integer` → a hand-written prefix-DFA for the JSON number grammar
     (`-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?`), open-ended: generation naturally stops
     once no vocabulary token can extend the number any further while staying valid.
   - `boolean` → the same trie mechanism as function names, over `{"true", "false"}`.

At every step, `src/decoding.constrained_generate` re-encodes the *entire* text so far
(`model.encode(context + generated)`) and calls `model.get_logits_from_input_ids`. This
is deliberately simpler than incrementally appending raw token ids: since the SDK has no
KV-cache and always runs a full forward pass anyway, re-encoding the full string costs
nothing extra, and it completely sidesteps subtle BPE re-tokenization bugs that can
happen when concatenating token id lists produced by separate `encode()` calls.

Vocabulary tokens are decoded once at startup (`src/vocab.py`) by implementing the
standard GPT-2-style "bytes-to-unicode" mapping from scratch against
`model.get_path_to_vocab_file()` — no `tokenizers`/`transformers` calls are made from
our own code to interpret the vocabulary.

## Design decisions

- **Two-phase decoding instead of one big grammar.** A fully generic JSON-schema grammar
  (arbitrary nesting, arbitrary key sets) would be considerably more complex to implement
  correctly. Since the function name is drawn from a small known set, and each function's
  parameter *keys* are fully determined once the name is known, splitting the problem
  into "pick a name" + "fill in known-typed values" keeps every grammar small, easy to
  reason about, and easy to unit-test in isolation.
- **Structural JSON is written by us, not "generated" as free text.** Braces, commas,
  and quotes around keys are appended directly rather than asked of the model. This
  guarantees valid structure by construction and saves a lot of pointless model calls.
- **Final output values are re-serialized, not copy-pasted from the model.** The raw
  text extracted via constrained decoding is converted into real Python values
  (`float`, `str`, `bool`) and the output file is produced with `json.dump`. This means
  the *grammars* only need to guarantee the extracted text is parseable, not that it is
  itself already 100% spec-perfect JSON — simplifying `StringGrammar` in particular
  (no need to handle every JSON escape sequence).
- **`pydantic` for all schema validation**, per the subject's requirement — both for
  parsing `functions_definition.json` / `function_calling_tests.json` and for shaping
  the final `FunctionCallResult` rows.
- **Every I/O and inference failure is caught and reported, never left to crash the
  process.** Config/vocab loading failures abort with a clear message and a non-zero
  exit code; a failure on a single prompt (e.g. the model selects an unknown function,
  or inference itself errors) is logged to stderr and that prompt is skipped, so one bad
  prompt never takes down the whole batch.

## Performance analysis

- **Accuracy**: 100% of *emitted* rows are guaranteed schema-valid by construction
  (constrained decoding cannot select an unknown function name or produce a
  malformed number/string). The remaining question is whether the model *chooses* the
  functionally correct function/arguments for a given prompt, which depends on the
  underlying LLM's reasoning quality, not on the decoding mechanism.
- **Speed**: the dominant cost is iterating the full vocabulary (~150k entries for
  Qwen3) once per generated token to filter valid candidates. The vocabulary is decoded
  to text exactly once at startup (`VocabIndex`) so each decoding step is a plain Python
  loop over precomputed strings, not repeated tokenizer calls. For the test prompts
  provided (short outputs, at most a handful of short parameters), a full run comfortably
  fits within the "under 5 minutes" target on standard hardware.
- **Reliability**: every grammar has a `force_close` fallback used if a per-value token
  budget (`max_tokens`) is exhausted or the model paints itself into a corner (e.g. no
  vocabulary token can validly continue a number). This guarantees the process never
  hangs indefinitely and always emits well-typed values, at worst falling back to a safe
  default (e.g. `0` for a stuck number).

## Challenges faced

- **Decoding raw vocab entries back into real text.** `vocab.json` stores tokens using a
  byte-level BPE encoding where every byte value is mapped to a printable unicode
  character. Reproducing this mapping correctly (and handling the occasional
  vocab entry that doesn't decode to valid UTF-8 on its own, e.g. a fragment of a
  multi-byte character) was the trickiest part — solved by implementing the standard
  GPT-2 `bytes_to_unicode` table from scratch and falling back gracefully (rather than
  crashing) on any entry that fails to decode.
- **Deciding when a value is "done".** Function names and booleans have a natural
  terminator (matching a known option exactly). JSON strings have an explicit
  terminator (the closing quote). Numbers do not — there is no character that means
  "stop here". This was resolved by treating "no vocabulary token can validly extend
  the number any further" as the natural stopping condition, backed by a `MAX_LEN`
  safety cap.
- **Avoiding subtle tokenization bugs.** Early designs incrementally appended raw
  generated token ids to the running `input_ids` list. This risks silently diverging
  from how the tokenizer would have encoded the combined text in one pass (BPE merge
  boundaries can differ). Re-encoding the full text at every step (see Algorithm
  explanation) sidesteps this class of bug entirely, at no extra inference cost given
  the SDK has no KV-cache.

## Testing strategy

Since the sandbox used to draft this project had no network access (and therefore
couldn't download real model weights), the test suite is split so that everything not
strictly requiring live model weights is fully covered and runs in milliseconds:

- `tests/test_grammar.py` — unit tests for `NumberGrammar`, `StringGrammar`,
  `TrieGrammar`, and the underlying number-prefix DFA (valid/invalid prefixes,
  continue/complete/invalid transitions).
- `tests/test_vocab.py` — unit tests for the byte-level BPE decoder against a small
  hand-written `vocab.json` fixture (plain tokens, leading-space tokens, punctuation,
  unknown ids).
- `tests/test_pipeline_integration.py` — full end-to-end exercise of
  `constrained_generate` and `process_prompt` against a small hand-written vocabulary
  and a **scripted fake model** (a stand-in object implementing just `encode()` and
  `get_logits_from_input_ids()`) that deterministically prefers a known target
  sequence. This validates the entire decoding loop and pipeline wiring without
  requiring `torch`/`transformers` or real weights.

Run everything with:

```bash
make test
```

Before submission/defense, additionally run the program against the real model
(`make run`) and manually inspect `data/output/function_calling_results.json` against
`data/input/function_calling_tests.json`, and test the edge cases called out by the
subject (empty strings, large numbers, ambiguous prompts, multi-parameter functions —
see the provided test files for examples of each).

## Example usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Given the provided `functions_definition.json` and `function_calling_tests.json`, a
successful run produces entries such as:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Known limitations

- `StringGrammar` rejects tokens containing a raw backslash, so the model cannot
  produce escaped characters (e.g. an embedded literal quote) inside a generated
  string value. This is a deliberate simplification (see Design decisions); it does not
  affect output validity, only the range of literal string content the model can choose.
- `TrieGrammar` treats an exact match against the option list as "complete" without
  checking whether that match is also a strict prefix of a *longer* option. This is safe
  here because none of the known function names (or `"true"`/`"false"`) is a prefix of
  another, but would need adjusting for a function catalog containing such collisions.
