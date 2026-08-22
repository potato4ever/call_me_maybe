# ABOUTME: Loads and validates the two JSON input files (functions + prompts).
# ABOUTME: Never raises raw json/pydantic exceptions; always wraps them in ConfigError.
try:
    import json
    from pathlib import Path
    from typing import Any, List

    from pydantic import ValidationError

    from .models import FunctionDefinition, PromptEntry
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
        file=sys.stderr,
    )


class ConfigError(Exception):
    """Raised when an input configuration file is missing, unreadable, or invalid."""

def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any]= {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_json_array(path: str) -> List[Any]:
    """Read *path* and return its parsed JSON content, guaranteed to be a list.

    Raises:
        ConfigError: if the file is missing, not valid JSON, or not a JSON array.
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
        path: path to a functions_definition.json file.

    Returns:
        The list of validated FunctionDefinition objects.

    Raises:
        ConfigError: on any I/O, JSON, or schema validation failure.
    """
    data = _load_json_array(path)
    if not data:
        raise ConfigError(f"{path} contains no function definitions")

    try:
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ConfigError(f"invalid function definition in {path}: {exc}") from exc

    names = [fn.name for fn in functions]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ConfigError(f"duplicate function name(s) in {path}: {sorted(duplicates)}")

    return functions


def load_prompts(path: str) -> List[PromptEntry]:
    """Load and validate the natural-language prompts file.

    Args:
        path: path to a function_calling_tests.json file.

    Returns:
        The list of validated PromptEntry objects.

    Raises:
        ConfigError: on any I/O, JSON, or schema validation failure.
    """
    data = _load_json_array(path)
    try:
        return [PromptEntry.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ConfigError(f"invalid prompt entry in {path}: {exc}") from exc
