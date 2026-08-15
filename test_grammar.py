import unittest

from src.grammar import (
    NumberGrammar,
    StringGrammar,
    TrieGrammar,
    _is_number_prefix,
)


class TestNumberPrefix(unittest.TestCase):
    def test_valid_prefixes(self) -> None:
        values = [
            "",
            "-",
            "0",
            "2",
            "20",
            "2.",
            "2.5",
            "-2.5",
            "2e",
            "2e-",
            "2e-3",
            "1e10",
            "3.14",
        ]

        for text in values:
            self.assertTrue(
                _is_number_prefix(text),
                text,
            )

    def test_invalid_prefixes(self) -> None:
        values = [
            "02",
            "01",
            "--2",
            "2..5",
            "2.e",
            "e5",
            ".5",
            "+5",
            "1e+",
        ]

        for text in values:
            self.assertFalse(
                _is_number_prefix(text),
                text,
            )


class TestNumberGrammar(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = NumberGrammar()

    def test_complete_numbers(self) -> None:
        for text in [
            "0",
            "12",
            "-12",
            "3.14",
            "0.5",
            "1e10",
            "-2.5E-3",
        ]:
            self.assertTrue(
                self.grammar.is_complete(text),
                text,
            )

    def test_incomplete_numbers(self) -> None:
        for text in [
            "-",
            "1.",
            "1e",
            "1e+",
        ]:
            self.assertFalse(
                self.grammar.is_complete(text),
                text,
            )

    def test_complete_token(self) -> None:
        self.assertEqual(
            self.grammar.check("", "12"),
            "complete",
        )

    def test_partial_token(self) -> None:
        self.assertEqual(
            self.grammar.check("", "-"),
            "continue",
        )

    def test_invalid_token(self) -> None:
        self.assertEqual(
            self.grammar.check("", "+5"),
            "invalid",
        )

    def test_force_close(self) -> None:
        self.assertEqual(
            self.grammar.force_close("-"),
            "0",
        )

        self.assertEqual(
            self.grammar.force_close("42"),
            "",
        )


class TestStringGrammar(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = StringGrammar()

    def test_plain_text(self) -> None:
        self.assertEqual(
            self.grammar.check("hel", "lo"),
            "continue",
        )

    def test_closing_quote(self) -> None:
        self.assertEqual(
            self.grammar.check("hello", '"'),
            "complete",
        )

    def test_unicode(self) -> None:
        self.assertEqual(
            self.grammar.check("", "héllo"),
            "continue",
        )

    def test_json_escape(self) -> None:
        self.assertEqual(
            self.grammar.check("hello", r"\""),
            "continue",
        )

    def test_unicode_escape(self) -> None:
        self.assertEqual(
            self.grammar.check("", r"\u"),
            "continue",
        )

        self.assertEqual(
            self.grammar.check("", r"\u1234"),
            "continue",
        )

    def test_invalid_control_character(self) -> None:
        self.assertEqual(
            self.grammar.check("hello", "\n"),
            "invalid",
        )

    def test_invalid_escape(self) -> None:
        self.assertEqual(
            self.grammar.check("hello", r"\q"),
            "invalid",
        )


class TestTrieGrammar(unittest.TestCase):
    def test_function_names(self) -> None:
        grammar = TrieGrammar(
            [
                "fn_greet",
                "fn_get_square_root",
            ]
        )

        self.assertEqual(
            grammar.check("", "fn_greet"),
            "complete",
        )

        self.assertEqual(
            grammar.check("fn_", "greet"),
            "complete",
        )

        self.assertEqual(
            grammar.check("fn_g", "oodbye"),
            "invalid",
        )

    def test_boolean_values(self) -> None:
        grammar = TrieGrammar(
            ["true", "false"]
        )

        self.assertEqual(
            grammar.check("", "true"),
            "complete",
        )

        self.assertEqual(
            grammar.check("", "tru"),
            "continue",
        )

        self.assertEqual(
            grammar.check("", "maybe"),
            "invalid",
        )


if __name__ == "__main__":
    unittest.main()
