"""Pydantic data models used across the project for schema and I/O validation.

These models provide automatic validation and clear error messages when
processing malformed function definitions or input prompts.
"""
import sys
try:
    from typing import Dict, Union, Set
    import keyword
    from pydantic import BaseModel, ConfigDict, field_validator
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
    )
    sys.exit(1)


#: Parameter/return types we know how to constrained-decode.
SUPPORTED_TYPES: Set[str] = {"number", "integer", "string", "boolean", "bool"}


class ParameterType(BaseModel):
    """Describes the JSON-schema type of a single parameter or return value.

    Attributes:
        type: The string identifier of the expected data type.
    """

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def _check_supported(cls, value: str) -> str:
        """Validate that the type is one supported by the decoder.

        Args:
            value: The type string to validate.

        Returns:
            str: The validated type string.

        Raises:
            ValueError: If the type is not in SUPPORTED_TYPES.
        """
        if value not in SUPPORTED_TYPES:
            raise ValueError(
                f"unsupported type {value!r}; "
                f"expected one of {sorted(SUPPORTED_TYPES)}"
            )
        return value


class FunctionDefinition(BaseModel):
    """A single callable function description from the schema.

    Attributes:
        name: The identifier of the function.
        description: A human-readable description of what the function does.
        parameters: A mapping of parameter names to their type definitions.
        returns: The expected return type definition.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        """Ensure the function name is a valid Python identifier.

        Args:
            value: The name string to validate.

        Returns:
            str: The validated name.

        Raises:
            ValueError: If name is not an identifier or is a reserved keyword.
        """
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError("invalid function name")
        return value


class PromptEntry(BaseModel):
    """A single natural-language request from the input file.

    Attributes:
        prompt: The raw text request to be processed.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt cannot be empty or whitespace-only")
        return stripped


class FunctionCallResult(BaseModel):
    """The final resolved function call for a specific prompt.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function selected by the LLM.
        parameters: The extracted arguments with their resolved values.
    """

    prompt: str
    name: str
    parameters: Dict[str, Union[float, int, str, bool]]
