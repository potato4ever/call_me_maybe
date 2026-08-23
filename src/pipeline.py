# ABOUTME: Orchestrates function selection and parameter generation.
# ABOUTME: Fixed JSON syntax is constructed by the program; the model only
# ABOUTME: generates semantic values.
try:
    from typing import Dict, List, Union

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
        file=sys.stderr,
    )


ParamValue = Union[float, int, str, bool]


class PipelineError(Exception):
    """Raised when a prompt cannot be converted into a valid function call."""


def _select_function(
    model: LLMModel,
    constraint_cache: ConstraintCache,
    functions: List[FunctionDefinition],
    functions_by_name: Dict[str, FunctionDefinition],
    prompt: str,
) -> tuple[FunctionDefinition, str]:

    selection_prompt = build_selection_prompt(
        functions,
        prompt,
    )


    grammar = TrieGrammar(
        [fn.name for fn in functions]
    )
    max_name_len = max(len(fn.name) for fn in functions)

    try:
        generated = constrained_generate(
            model,
            selection_prompt,
            grammar,
            constraint_cache=constraint_cache,
            max_tokens=24,
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

    context = ( f'Request: "{prompt}"\n' 
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
