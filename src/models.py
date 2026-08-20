# ABOUTME: Pydantic data models used across the project (schema + I/O validation).
# ABOUTME: These models give us free validation and clear error messages on malformed input.

from typing import Dict, Union
import keyword
from pydantic import BaseModel, ConfigDict, field_validator

#: Parameter/return types we know how to constrained-decode.
SUPPORTED_TYPES = {"number", "integer", "string", "boolean", "bool"}


class ParameterType(BaseModel):
    """Describes the JSON-schema-like type of a single parameter or return value."""

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def _check_supported(cls, value: str) -> str:
        if value not in SUPPORTED_TYPES:
            raise ValueError(
                f"unsupported type {value!r}; expected one of {sorted(SUPPORTED_TYPES)}"
            )
        return value


class FunctionDefinition(BaseModel):
    """A single callable function, as described in functions_definition.json."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType
    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError("invalid functoin name")
        return value


class PromptEntry(BaseModel):
    """A single natural-language request, as described in function_calling_tests.json."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """A single row of the output file: the resolved function call for one prompt."""

    prompt: str
    name: str
    parameters: Dict[str, Union[float, int, str, bool]]
