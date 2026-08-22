# ABOUTME: Builds the natural-language context given to the model before
# ABOUTME: constrained decoding selects the function.
try:
    from typing import List

    from .models import FunctionDefinition
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
        file=sys.stderr,
    )


def _describe_function(fn: FunctionDefinition) -> str:
    params = ", ".join(
        f"{name}: {ptype.type}"
        for name, ptype in fn.parameters.items()
    )

    return (
        f"- {fn.name}({params}): "
        f"{fn.description}"
    )


def build_selection_prompt(
    functions: List[FunctionDefinition],
    user_prompt: str,
) -> str:
    function_list = "\n".join(
        _describe_function(fn)
        for fn in functions
    )

    return (
        "You are a function calling assistant. Read the user's "
        "request and choose exactly one function from the list "
        "that best fulfils it.\n\n"
        f"Available functions:\n{function_list}\n\n"
        f'User request: "{user_prompt}"\n\n'
        "Respond with a single JSON object describing the "
        "function call.\n"
        'Function call: {"name": "'
    )
