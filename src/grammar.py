# ABOUTME: Small grammar/state-machine objects used by constrained decoding.
# ABOUTME: They validate generated text and report whether a candidate continues
# ABOUTME: or completes the current semantic value.
try:
    import re
    from typing import List, Protocol
except Exception as exc:
    print(
        f"Error: could not import required module: {exc}",
        file=sys.stderr,
    )

GrammarStatus = str


class Grammar(Protocol):
    def check(self, so_far: str, token: str) -> GrammarStatus:
        ...

    def is_complete(self, so_far: str) -> bool:
        ...

    def force_close(self, so_far: str) -> str:
        ...

    def consume_completion(self, so_far: str, token: str) -> str:
        ...


class StringGrammar:
    r"""Grammar for the content of a JSON string.

    The caller emits the opening quote. This grammar is responsible for
    generating the string content and the final closing quote.

    Valid JSON escapes are supported:
        \"
        \\
        \/
        \b
        \f
        \n
        \r
        \t
        \uXXXX
    """

    MAX_LEN = 200

    _ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t"}

    def consume_completion(self, so_far: str, token: str) -> str:
        """Return only the semantic part of a completion token.

        The decoder may select a token containing the closing quote and
        additional JSON syntax, for example:

            '"}}\n'

        Only the closing quote belongs to this string value. Everything
        after it belongs to the surrounding JSON structure and is discarded.
        """
        quote_index = token.find('"')

        if quote_index != -1:
            return token[:quote_index + 1]

        return token

    def check(self, so_far: str, token: str) -> GrammarStatus:
        if not token:
            return "invalid"

        candidate = so_far + token

        if len(candidate) > self.MAX_LEN:
            return "invalid"

        state = self._validate(candidate)

        if state == "complete":
            return "complete"

        if state == "prefix":
            return "continue"

        return "invalid"

    def is_complete(self, so_far: str) -> bool:
        if not so_far.endswith('"'):
            return False

        return self._validate(so_far) == "complete"

    def force_close(self, so_far: str) -> str:
        if self._validate(so_far) == "complete":
            return ""

        # If an escape is unfinished, complete it safely.
        if so_far.endswith("\\"):
            return '"'

        if "\\u" in so_far:
            index = so_far.rfind("\\u")
            digits = so_far[index + 2:]

            if 0 < len(digits) < 4 and all(
                char in "0123456789abcdefABCDEF"
                for char in digits
            ):
                return "0" * (4 - len(digits)) + '"'

        return '"'

    @classmethod
    def _validate(cls, text: str) -> GrammarStatus:
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
    """Return whether text is a valid, possibly incomplete JSON number."""

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
        """
        Return the semantic value portion of a token that completes
        the number.

        Example:
            so_far='2.0', token=',' -> ''
            so_far='2.0', token='}' -> ''
        """
        candidate = so_far + token

        number, delimiter = self._split_number_and_delimiter(candidate)

        if number is not None and delimiter is not None:
            if self.is_complete_number(number):
                return number[len(so_far):]

        return token

    def check(self, so_far: str, token: str) -> GrammarStatus:
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
        """A number is complete only after its JSON delimiter appears."""
        number, delimiter = self._split_number_and_delimiter(text)

        return (
            number is not None
            and delimiter is not None
            and self.is_complete_number(number)
        )

    def force_close(self, so_far: str) -> str:
        """
        Finish a valid number with a JSON delimiter.

        This is only a fallback. Normally the model should generate
        the delimiter itself.
        """
        return ""

    def is_valid_prefix(self, text: str) -> bool:
        """Return True if text can still become a JSON number."""
        return _is_number_prefix(text)

    def is_complete_number(self, text: str) -> bool:
        """Return True only for a complete JSON number."""
        return bool(_NUMBER_RE.fullmatch(text))

    def is_delimiter(self, text: str) -> bool:
        return text in self.JSON_DELIMITERS

    def _split_number_and_delimiter(self, text: str):
        """
        Split a number followed by optional JSON whitespace and
        one JSON delimiter.

        Examples:
            '265,'    -> ('265', ',')
            '345}'    -> ('345', '}')
            '1.25]'   -> ('1.25', ']')
            '2.0 ,'   -> ('2.0', ',')
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

    def __init__(self, options: List[str]):
        if not options:
            raise ValueError("TrieGrammar requires at least one option")

        self._options = tuple(options)

    def consume_completion(self, so_far: str, token: str) -> str:
        return token

    def check(self, so_far: str, token: str) -> GrammarStatus:
        candidate = so_far + token
        for option in self._options:
            if candidate.endswith("\""):
                if candidate[:-1] in self._options:
                    return "complete"
                else:
                    return "invalid"
        return "continue"

    def is_complete(self, so_far: str) -> bool:
        return (
            so_far.endswith('"')
            and so_far[:-1] in self._options
        )

    def force_close(self, so_far: str) -> str:
        for option in self._options:
            if option.startswith(so_far):
                return option[len(so_far):]

        return ""
