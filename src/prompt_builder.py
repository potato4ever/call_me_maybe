"""Builds the natural-language context used for function selection.

This module provides utilities to format function definitions into a list
and construct the final system prompt that guides the LLM to choose the
correct function name.
"""
import sys
try:
    from typing import List

    from .models import FunctionDefinition
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
    )
    sys.exit(1)


def _describe_function(fn: FunctionDefinition) -> str:
    """Format a single function definition into a readable string summary.

    Args:
        fn: The function definition object to format.

    Returns:
        str: A string in the format "- name(params): description".
    """
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
    """Construct the full text prompt for the LLM function selection phase.

    Args:
        functions: A list of all available function definitions.
        user_prompt: The raw natural language request from the user.

    Returns:
        str: The complete prompt ending with the start of the JSON response.
    """
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
