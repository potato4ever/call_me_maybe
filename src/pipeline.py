"""Orchestrates function selection and parameter generation.

This module provides the high-level logic for choosing which function
to call based on a natural language prompt and then generating each
required parameter using constrained decoding.
"""
import sys
try:
    from typing import Dict, List, Union, Tuple

    from .decoding import (
        ConstraintCache,
        DecodingError,
        LLMModel,
        constrained_generate,
    )
    from .grammar import NumberGrammar, StringGrammar, TrieGrammar
    from .models import FunctionCallResult, FunctionDefinition
    from .prompt_builder import build_selection_prompt
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
    )
    sys.exit(1)


ParamValue = Union[float, int, str, bool]


class PipelineError(Exception):
    """Raised when a prompt cannot be converted into a valid function call."""


def _select_function(
    model: LLMModel,
    constraint_cache: ConstraintCache,
    functions: List[FunctionDefinition],
    functions_by_name: Dict[str, FunctionDefinition],
    prompt: str,
) -> Tuple[FunctionDefinition, str]:
    """Use the LLM to select which function best matches the user prompt.

    Args:
        model: The LLM model instance.
        constraint_cache: Cache for token tries.
        functions: List of all available function definitions.
        functions_by_name: Mapping of names to definitions for quick lookup.
        prompt: The user's natural language request.

    Returns:
        Tuple[FunctionDefinition, str]: The selected function definition
            and the updated prompt context for parameter generation.

    Raises:
        PipelineError: If function selection fails or returns an unknown name.
    """

    selection_prompt = build_selection_prompt(
        functions,
        prompt,
    )
    grammar = TrieGrammar(
        [fn.name for fn in functions]
    )
    max_name_len = max([len(fn.name) for fn in functions])

    try:
        generated = constrained_generate(
            model,
            selection_prompt,
            grammar,
            constraint_cache=constraint_cache,
            max_tokens=max_name_len,
        )

    except DecodingError as exc:
        raise PipelineError(
            f"function selection failed: {exc}"
        ) from exc

    if generated not in functions_by_name:
        raise PipelineError(
            f"model selected unknown function {generated!r}"
        )

    context = (
        selection_prompt
        + generated
        + '", '
    )

    return functions_by_name[generated], context


def _generate_parameter_value(
    model: LLMModel,
    constraint_cache: ConstraintCache,
    prompt: str,
    function_name: str,
    function_description: str,
    param_name: str,
    param_type: str,
    parameters: Dict[str, ParamValue],
) -> ParamValue:
    """Generate a single parameter value using the appropriate grammar.

    Args:
        model: The LLM model instance.
        constraint_cache: Cache for token tries.
        prompt: The original user request.
        function_name: Name of the selected function.
        function_description: Description of the selected function.
        param_name: The name of the parameter to generate.
        param_type: The expected type ('string', 'number', 'bool', etc.).
        parameters: Dictionary of parameters already generated.

    Returns:
        ParamValue: The generated value cast to the correct Python type.

    Raises:
        DecodingError: If generation fails for the specific type.
        PipelineError: If the parameter type is unsupported.
    """

    context = (f'Request: "{prompt}"\n'
               f"Function: {function_name} — {function_description}\n"
               f"Parameters so far: {parameters!r}\n")
    for name, value in parameters.items():
        context += f'"{name}": {value!r}, '

    # ---------------------------------------------------------------
    # String parameter
    # ---------------------------------------------------------------
    if param_type == "string":
        context += f'"{param_name}": "'

        generated = constrained_generate(
            model,
            context,
            StringGrammar(),
            constraint_cache=constraint_cache,
            max_tokens=100,
        )
        status = StringGrammar().validate(generated)
        if status == "invalid" or status == "prefix":
            raise DecodingError("parameters selection failed")
        text_value = (
            generated[:-1]
            if generated.endswith('"')
            else generated
        )
        return text_value

    # ---------------------------------------------------------------
    # Number parameter
    # ---------------------------------------------------------------
    if param_type in ("number", "integer"):
        context += f'"{param_name}": '
        generated = constrained_generate(
            model,
            context,
            NumberGrammar(),
            constraint_cache=constraint_cache,
            max_tokens=24,
        )

        if param_type == "integer":
            try:
                value = int(generated)
            except ValueError as exc:
                raise DecodingError(
                    f"invalid integer generated: {generated!r}"
                ) from exc

            return value

        try:
            value = float(generated)
        except ValueError as exc:
            raise DecodingError(
                f"invalid number generated: {generated!r}"
            ) from exc

        return value

    # ---------------------------------------------------------------
    # Boolean parameter
    # ---------------------------------------------------------------
    if param_type in ("boolean", "bool"):
        context += f'"{param_name}": '
        generated = constrained_generate(
            model,
            context,
            TrieGrammar(["true", "false"]),
            constraint_cache=constraint_cache,
            max_tokens=8,
        )

        return generated == "true"

    raise PipelineError(
        f"unsupported parameter type: {param_type!r}"
    )


def process_prompt(
    model: LLMModel,
    constraint_cache: ConstraintCache,
    functions_by_name: Dict[str, FunctionDefinition],
    functions: List[FunctionDefinition],
    prompt: str,
) -> FunctionCallResult:
    """Orchestrate the full transformation from prompt to function call result.

    Args:
        model: The LLM model instance.
        constraint_cache: Cache for token tries.
        functions_by_name: Dictionary for fast function lookup.
        functions: List of all function definitions for the selection prompt.
        prompt: The natural language request.

    Returns:
        FunctionCallResult: A validated Pydantic model containing the prompt,
            function name, and extracted parameters.

    Raises:
        PipelineError: If selection or parameter extraction fails.
    """

    fn_def, _ = _select_function(
        model,
        constraint_cache,
        functions,
        functions_by_name,
        prompt,
    )

    parameters: Dict[str, ParamValue] = {}

    for param_name, param_type in fn_def.parameters.items():

        try:
            value = _generate_parameter_value(
                model,
                constraint_cache,
                prompt,
                fn_def.name,
                fn_def.description,
                param_name,
                param_type.type,
                parameters,
            )

        except DecodingError as exc:
            raise PipelineError(
                f"parameter {param_name!r} generation failed: {exc}"
            ) from exc

        parameters[param_name] = value

    return FunctionCallResult(
        prompt=prompt,
        name=fn_def.name,
        parameters=parameters,
    )
