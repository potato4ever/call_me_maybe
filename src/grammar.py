"""Small grammar/state-machine objects used by constrained decoding.

This module provides the validation logic for different JSON types (strings,
numbers, and literal options) to ensure generated tokens remain valid.
"""
import sys
try:
    import re
    from typing import List, Protocol, Tuple, Optional
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}"
    )
    sys.exit(1)

GrammarStatus = str


class Grammar(Protocol):
    """Protocol for grammar objects used in constrained decoding."""
    def check(self, so_far: str, token: str) -> GrammarStatus:
        """Check if adding a token to the current string is valid.

        Args:
            so_far: The string generated up to this point.
            token: The new token candidate to evaluate.

        Returns:
            GrammarStatus: 'invalid', 'continue', or 'complete'.
        """
        ...

    def is_complete(self, so_far: str) -> bool:
        """Determine if the current string represents a full valid value.

        Args:
            so_far: The string to evaluate.

        Returns:
            bool: True if the grammar is satisfied and closed.
        """
        ...

    def consume_completion(self, so_far: str, token: str) -> str:
        """Extract only the relevant part of
        a token that completes the grammar.

        Args:
            so_far: The string generated so far.
            token: The token that finishes the value.

        Returns:
            str: The part of the token belonging to this grammar.
        """
        ...


class StringGrammar:
    r"""Grammar for the content of a JSON string.

    The caller emits the opening quote. This grammar is responsible for
    generating the string content and the final closing quote.

    Valid JSON escapes are supported: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX.
    """

    MAX_LEN = 200

    _ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t"}

    def consume_completion(self, so_far: str, token: str) -> str:
        """Return only the semantic part of a completion token.

        Args:
            so_far: String generated so far.
            token: Token potentially containing extra JSON syntax.

        Returns:
            str: The portion of the token up to the closing quote.
        """
        quote_index = token.find('"')

        if quote_index != -1:
            return token[:quote_index + 1]

        return token

    def check(self, so_far: str, token: str) -> GrammarStatus:
        """Validate if a token continues or completes a JSON string.

        Args:
            so_far: Current generated content.
            token: New token to evaluate.

        Returns:
            GrammarStatus: State of the grammar.
        """
        if not token:
            return "invalid"

        candidate = so_far + token

        if len(candidate) > self.MAX_LEN:
            return "invalid"

        state = self.validate(candidate)

        if state == "complete":
            return "complete"

        if state == "prefix":
            return "continue"

        return "invalid"

    def is_complete(self, so_far: str) -> bool:
        """Check if the string is finished and valid.

        Args:
            so_far: The string to check.

        Returns:
            bool: True if the string is complete.
        """
        if not so_far.endswith('"'):
            return False

        return self.validate(so_far) == "complete"

    @classmethod
    def validate(cls, text: str) -> GrammarStatus:
        """State machine to validate JSON string escape sequences and length.

        Args:
            text: The text to validate.

        Returns:
            GrammarStatus: The internal state.
        """
        escaped = False
        unicode_digits = 0

        for char in text:
            if unicode_digits:
                if char not in "0123456789abcdefABCDEF":
                    return "invalid"

                unicode_digits += 1

                if unicode_digits == 4:
                    unicode_digits = 0
                    escaped = False

                continue

            if escaped:
                if char == "u":
                    unicode_digits = 1
                    continue

                if char not in cls._ESCAPE_CHARS:
                    return "invalid"

                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                return "complete"

            if ord(char) < 0x20:
                return "invalid"

        if escaped or unicode_digits:
            return "prefix"

        return "prefix"


def _is_number_prefix(text: str) -> bool:
    """Return whether text is a valid, possibly incomplete JSON number.

    Args:
        text: The string to evaluate.

    Returns:
        bool: True if it is a valid numeric prefix.
    """

    if text == "":
        return True

    i = 0
    n = len(text)

    if text[i] == "-":
        i += 1

        if i == n:
            return True

    if i == n:
        return True

    if text[i] == "0":
        i += 1

        # JSON numbers cannot have another integer digit after zero.
        if i < n and text[i].isdigit():
            return False

    elif text[i] in "123456789":
        i += 1

        while i < n and text[i].isdigit():
            i += 1

    else:
        return False

    if i == n:
        return True

    if text[i] == ".":
        i += 1

        if i == n:
            return True

        if not text[i].isdigit():
            return False

        while i < n and text[i].isdigit():
            i += 1

        if i == n:
            return True

    if i < n and text[i] in "eE":
        i += 1

        if i < n and text[i] in "+-":
            i += 1
        if i == n:
            return True

        if not text[i].isdigit():
            return False

        while i < n and text[i].isdigit():
            i += 1

    return i == n


_NUMBER_RE = re.compile(
    r"^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$"
)


class NumberGrammar:
    """Grammar for a JSON number terminated by a JSON delimiter."""

    JSON_DELIMITERS = {",", "}", "]"}

    def consume_completion(self, so_far: str, token: str) -> str:
        """Return the numeric portion of a completing token.

        Args:
            so_far: The numeric string generated so far.
            token: The token that finishes the number.

        Returns:
            str: The semantic numeric value.
        """
        candidate = so_far + token

        number, delimiter = self._split_number_and_delimiter(candidate)

        if number is not None and delimiter is not None:
            if self.is_complete_number(number):
                return number[len(so_far):]

        return token

    def check(self, so_far: str, token: str) -> GrammarStatus:
        """Validate numeric tokens against JSON rules and delimiters.

        Args:
            so_far: Current numeric string.
            token: Next token candidate.

        Returns:
            GrammarStatus: Validation state.
        """
        if not token:
            return "invalid"

        candidate = so_far + token

        # The candidate is still entirely inside the number.
        if self.is_valid_prefix(candidate):
            return "continue"

        # The candidate may finish the number and immediately contain
        # a JSON delimiter, for example:
        #
        #     "123,"
        #     "123}"
        #     "1.25]"
        #
        number, delimiter = self._split_number_and_delimiter(candidate)

        if (
            number is not None
            and delimiter is not None
            and self.is_complete_number(number)
        ):
            return "complete"

        return "invalid"

    def is_complete(self, text: str) -> bool:
        """Determine if numeric generation is finished.

        Args:
            text: The generated string.

        Returns:
            bool: True if the number is complete with a delimiter.
        """
        number, delimiter = self._split_number_and_delimiter(text)

        return (
            number is not None
            and delimiter is not None
            and self.is_complete_number(number)
        )

    def is_valid_prefix(self, text: str) -> bool:
        """Check if text is a valid numeric start.

        Args:
            text: Prefix to check.

        Returns:
            bool: True if valid prefix.
        """
        return _is_number_prefix(text)

    def is_complete_number(self, text: str) -> bool:
        """Validate a full JSON number against regex.

        Args:
            text: Number string to validate.

        Returns:
            bool: True if complete JSON number.
        """
        return bool(_NUMBER_RE.fullmatch(text))

    def is_delimiter(self, text: str) -> bool:
        """Check if character is a JSON delimiter.

        Args:
            text: Char to check.

        Returns:
            bool: True if delimiter.
        """
        return text in self.JSON_DELIMITERS

    def _split_number_and_delimiter(self, text: str) -> Tuple[
                                            Optional[str],
                                            Optional[str]]:
        """Split a string into a number and its trailing JSON delimiter.

        Args:
            text: The candidate string.

        Returns:
            Tuple[Optional[str], Optional[str]]: The split components.
        """

        text = text.rstrip()

        if len(text) < 2:
            return None, None

        delimiter = text[-1]

        if delimiter not in self.JSON_DELIMITERS:
            return None, None

        number = text[:-1].rstrip()

        return number, delimiter


class TrieGrammar:
    """Grammar for a fixed set of complete literal values."""

    def __init__(self, options: List[str]) -> None:
        """Initialize the trie grammar with valid options.

        Args:
            options: List of allowed string values.

        Raises:
            ValueError: If options is empty.
        """
        if not options:
            raise ValueError("TrieGrammar requires at least one option")

        self._options = tuple(options)

    def consume_completion(self, so_far: str, token: str) -> str:
        """Return the completion token.

        Args:
            so_far: Current string.
            token: Completion token.

        Returns:
            str: The token itself.
        """
        return token

    def check(self, so_far: str, token: str) -> GrammarStatus:
        """Check if a token leads to a valid entry in the trie.

        Args:
            so_far: Current string.
            token: Token to append.

        Returns:
            GrammarStatus: Validation state.
        """
        candidate = so_far + token
        for option in self._options:
            if candidate.endswith("\""):
                if candidate[:-1] in self._options:
                    return "complete"
                else:
                    return "invalid"
        return "continue"

    def is_complete(self, so_far: str) -> bool:
        """Check if string is a valid quoted entry.

        Args:
            so_far: The string to check.

        Returns:
            bool: True if valid.
        """
        return (
            so_far.endswith('"')
            and so_far[:-1] in self._options
        )
