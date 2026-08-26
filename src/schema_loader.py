"""Loads and validates the two JSON input files (functions + prompts).

This module ensures that input files are valid JSON, contain the correct
data structures, and adhere to the project's specific schema.
"""
import sys
try:
    import json
    from pathlib import Path
    from typing import Any, List, Tuple, Dict

    from pydantic import ValidationError

    from .models import FunctionDefinition, PromptEntry
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
    )
    sys.exit(1)


class ConfigError(Exception):
    """Raised when an input configuration file is missing,
    unreadable, or invalid.
    """


def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Object pairs hook for json.loads to reject duplicate keys in JSON.

    Args:
        pairs: A list of key-value tuples from the JSON parser.

    Returns:
        Dict[str, Any]: A dictionary if no duplicates are found.

    Raises:
        ValueError: If a duplicate key is encountered.
    """
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_json_array(path: str) -> List[Any]:
    """Read path and return its parsed JSON content, guaranteed to be a list.

    Args:
        path: Path to the JSON file.

    Returns:
        List[Any]: The parsed JSON list content.

    Raises:
        ConfigError: If the file is missing, not valid JSON, or not an array.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"file not found: {path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ConfigError(f"{path} must contain a JSON array at the top level")

    return data


def load_functions_definition(path: str) -> List[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Path to a functions_definition.json file.

    Returns:
        List[FunctionDefinition]: The list of validated
        FunctionDefinition objects.

    Raises:
        ConfigError: On any I/O, JSON, or schema validation failure.
    """
    data = _load_json_array(path)
    if not data:
        raise ConfigError(f"{path} contains no function definitions")

    try:
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ConfigError(f"invalid function definition in {path}: {exc}")

    names = [fn.name for fn in functions]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ConfigError(f"duplicate function name(s) in {path}: "
                          f"{sorted(duplicates)}")

    return functions


def load_prompts(path: str) -> List[PromptEntry]:
    """Load and validate the natural-language prompts file.

    Args:
        path: Path to a function_calling_tests.json file.

    Returns:
        List[PromptEntry]: The list of validated PromptEntry objects.

    Raises:
        ConfigError: On any I/O, JSON, or schema validation failure.
    """
    data = _load_json_array(path)
    try:
        return [PromptEntry.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ConfigError(f"invalid prompt entry in {path}: {exc}") from exc
