# ABOUTME: Command-line entry point for call_me_maybe.
# ABOUTME: Loads configuration and model, prepares token constraints once,
# ABOUTME: processes prompts, and writes validated results.

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from .decoding import ConstraintCache
from .pipeline import PipelineError, process_prompt
from .schema_loader import (
    ConfigError,
    load_functions_definition,
    load_prompts,
)


DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calling_results.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description=(
            "Translate natural language prompts into structured "
            "function calls."
        ),
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_PATH,
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        functions = load_functions_definition(
            args.functions_definition
        )
        prompts = load_prompts(args.input)

    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    functions_by_name = {
        fn.name: fn
        for fn in functions
    }

    try:
        from llm_sdk.llm_sdk import Small_LLM_Model

        model = Small_LLM_Model(
            model_name=args.model
        )

    except Exception as exc:
        print(
            f"Error: could not load model {args.model!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Preparing token constraints...")

    try:
        constraint_cache = ConstraintCache(model)

    except Exception as exc:
        print(
            f"Error: could not prepare token constraints: {exc}",
            file=sys.stderr,
        )
        return 1

    results: List[dict] = []

    for entry in prompts:
        try:
            result = process_prompt(
                model,
                constraint_cache,
                functions_by_name,
                functions,
                entry.prompt,
            )

        except PipelineError as exc:
            print(
                f"Warning: skipping prompt {entry.prompt!r}: {exc}",
                file=sys.stderr,
            )
            continue

        except Exception as exc:
            print(
                f"Warning: unexpected error on prompt "
                f"{entry.prompt!r}: {exc}",
                file=sys.stderr,
            )
            continue

        print(result)
        results.append(result.model_dump())

    output_path = Path(args.output)

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                results,
                handle,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as exc:
        print(
            f"Error: could not write output file "
            f"{output_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Wrote {len(results)}/{len(prompts)} "
        f"result(s) to {output_path}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
