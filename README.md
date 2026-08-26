*This project has been created as part of the 42 curriculum by zabelhac.*

# call me maybe — Introduction to function calling in LLMs

## Description

This project turns natural-language requests ("What is the sum of 2 and 3?") into
structured, machine-executable function calls (`{"name": "fn_add_numbers", "parameters":
{"a": 2.0, "b": 3.0}}`), using a small (0.6B parameter) local LLM, **Qwen/Qwen3-0.6B**.

Small models are often unreliable at spontaneously producing valid JSON. Instead of
relying only on prompting, this project implements **constrained decoding**: at every
generation step, the model's output probabilities (logits) are used to select only
tokens that keep the generated value valid according to the expected grammar and
function schema.

The result is guaranteed to follow the supported output structure and parameter
types, regardless of the model's raw reliability. However, constrained decoding does
not guarantee that the model chooses the functionally correct function or extracts
the correct values from the user's request.

The program reads:
- `data/input/functions_definition.json` — the catalog of callable functions (name,
  parameter types, description).
- `data/input/function_calling_tests.json` — a list of natural-language prompts.

...and writes `data/output/function_calling_results.json`: for every successfully
processed prompt, the selected function name and its correctly-typed arguments.

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

1. Function selection. The model is prompted with the list of available functions (name, parameter types, description) and the user's request, ending at {"name": ". From there, decoding is constrained to a token trie built from the set of known function names. At each step, only tokens that continue a valid function-name prefix are considered, and the token with the highest raw logit among those allowed tokens is selected. Generation stops when the generated function name is complete.
2. Parameter generation. Once the function is selected, its parameter schema is known. Structural JSON tokens such as {, "key": , ,, }, and opening quotes are constructed directly by the pipeline rather than generated by the model. Only parameter values are generated by the model and constrained according to their declared type:
- `string` → generated using StringGrammar, which validates JSON string content and escape sequences and stops when the closing quote is generated.
- `number / integer` → generated using NumberGrammar, which implements a hand-written prefix-DFA for the JSON number grammar (-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?). Generation continues while the text is a valid number prefix and completes when a valid JSON delimiter is encountered.
- `boolean` → generated using the same token-trie mechanism as function names, with the valid options true and false.

At every generation step, src/decoding.constrained_generate re-encodes the complete context and generated text using model.encode() and obtains the next-token logits using model.get_logits_from_input_ids(). The decoder then restricts the candidate tokens according to the active grammar and selects the highest-logit valid token.

For closed grammars such as function names and booleans, the token trie is built once and cached by ConstraintCache, avoiding repeated tokenization of the same valid values. For dynamic grammars such as strings and numbers, vocabulary tokens are checked against the grammar directly.

The vocabulary is loaded from the model's vocabulary JSON file through model.get_path_to_vocab_file(). The vocabulary IDs are converted into a NumPy array and used as the candidate token IDs during dynamic constrained decoding.


## Design decisions
- Two-phase decoding instead of one big grammar. Function selection and parameter generation are handled separately. Once the function is selected, its parameter names and types are already known, which keeps the grammars simpler and easier to test.
- Structural JSON is written by the pipeline. Braces, commas, parameter names, and other deterministic JSON structure are constructed directly instead of being generated by the model. This guarantees the structure and avoids unnecessary model calls.
- Trie-based constraints for closed values. Function names and boolean values come from a known finite set, so a token trie is used to restrict generation efficiently. ConstraintCache reuses tries for repeated sets of options.
- Grammar-based constraints for dynamic values. Strings and numbers cannot be represented by a small fixed list of values, so they are validated token by token using StringGrammar and NumberGrammar.
- Final values are converted to Python types. Generated values are converted to str, int, float, or bool before being stored in FunctionCallResult. The final results are then serialized with json.dump.
- Pydantic for schema validation. Pydantic models validate the input function definitions and prompts, and also validate the final function-call result.
Error handling. Configuration and model-loading failures terminate with an error, while failures during an individual prompt are reported and that prompt is skipped so the remaining prompts can still be processed.

## Performance analysis

- **Accuracy**: 100% of *emitted* rows are guaranteed to follow the supported
  schema by construction. Constrained decoding cannot select an unknown function
  name or produce an invalid value for the supported parameter types. However,
  constrained decoding does not guarantee that the model chooses the functionally
  correct function or arguments for a given prompt. This depends on the underlying
  LLM's reasoning ability.

- **Speed**: the main computational cost is evaluating vocabulary tokens during
  dynamic decoding. For strings and numbers, the vocabulary is checked at each
  generated token to determine which tokens are valid according to the grammar.
  Closed grammars such as function names and booleans use cached token tries,
  avoiding repeated construction of the same constraints.

- **Inference cost**: at every generation step, the complete context and generated
  text are re-encoded and passed through the model to obtain the next-token logits.
  This simplifies tokenization handling, but requires a model inference for every
  generated token.

- **Reliability**: each generation has a `max_tokens` limit, preventing an
  individual generation from running indefinitely. If no valid token can continue
  the grammar, a `DecodingError` is raised and the affected prompt is skipped.

## Challenges faced

- **Constrained token generation:** One of the main challenges was restricting the
  model to only generate tokens that keep the output valid. This was solved by
  implementing a token trie for fixed values such as function names and booleans,
  and dedicated grammar state machines for strings and numbers.

- **Number completion:** Unlike strings, numbers do not have an explicit closing
  character. The `NumberGrammar` therefore uses a prefix-DFA to distinguish valid
  number prefixes from complete numbers and detects completion when a JSON
  delimiter is encountered.

- **String validation:** JSON strings can contain escape sequences and control
  characters, so `StringGrammar` implements validation for supported JSON escape
  sequences while detecting the closing quote.

- **Tokenization consistency:** Generated text is re-encoded at every step instead
  of manually concatenating token IDs. This ensures that the model receives the
  same tokenization it would get when encoding the complete generated text.

- **Balancing correctness and simplicity:** Instead of implementing a complete
  general-purpose JSON Schema grammar, the project uses a two-phase approach:
  function selection followed by parameter generation. This keeps the individual
  constraints smaller and easier to test while still enforcing the supported
  function-calling schema.

## Testing strategy

The project was tested at different levels to verify both individual components
and the complete function-calling pipeline.

- **Grammar testing:** Valid and invalid JSON strings, escape sequences, number
  prefixes, complete numbers, delimiters, and fixed-value options were tested.

- **Constrained decoding testing:** The decoder was tested to ensure that invalid
  tokens are rejected and that only tokens allowed by the active grammar can be
  selected.

- **Input validation testing:** Invalid JSON files, malformed function
  definitions,
  unsupported parameter types, duplicate function names, invalid function names,
  and empty prompts were tested.

- **Pipeline testing:** Function selection and parameter generation were tested
  together using controlled model outputs to verify that the different components
  work correctly together.

- **Error handling testing:** Missing files, invalid JSON, invalid vocabulary
  data, model inference failures, and cases where no valid token can be generated
  were tested to ensure errors are handled without crashing the entire batch.

- **End-to-end testing:** The complete pipeline was tested to verify that
  natural-language prompts are converted into correctly structured and typed
  function calls.
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

- `NumberGrammar` requires a JSON delimiter to detect that a number is complete.
  Therefore, number generation depends on the surrounding JSON structure providing
  a valid delimiter such as `,`, `}`, or `]`.

- Generated string values are limited to 200 characters by
  `StringGrammar.MAX_LEN`.

- The decoder evaluates vocabulary tokens at each generation step. For dynamic
  grammars such as strings and numbers, this can be computationally expensive
  because many vocabulary tokens may need to be checked for every generated token.

- The current implementation supports a fixed set of parameter types
  (`string`, `number`, `integer`, `boolean`, and `bool`). More complex JSON types,
  such as arrays, objects, and nested schemas, are not currently supported.
