"""Command-line entry point for the call_me_maybe project.

This module coordinates the loading of configuration,
initialization of the LLM,
processing of natural language prompts through a constrained decoding pipeline,
and saving the validated results to a JSON file.
"""
import sys
try:
    import argparse
    import json
    from pathlib import Path
    from typing import List, Dict, Sequence, Any

    from .decoding import ConstraintCache
    from .pipeline import PipelineError, process_prompt
    from .schema_loader import (
        ConfigError,
        load_functions_definition,
        load_prompts,
    )
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calling_results.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of command-line arguments. Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace: The parsed arguments containing file paths.
    """
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description=(
            "Translate natural language "
            "prompts into structured "
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main execution logic for the function calling tool.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)

    try:
        functions = load_functions_definition(
            args.functions_definition
        )
        prompts = load_prompts(args.input)

    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    functions_by_name: Dict[str, Any] = {
        fn.name: fn
        for fn in functions
    }

    try:
        from llm_sdk.llm_sdk import Small_LLM_Model

        model = Small_LLM_Model()

    except Exception as exc:
        print(
            f"Error: could not load model: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Preparing token constraints...")

    constraint_cache = ConstraintCache(model)

    results: List[Dict[str, Any]] = []

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
    try:
        main()
    except KeyboardInterrupt:
        print("exit")
    except Exception as e:
        print(f"Error: {e}")
