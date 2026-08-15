.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	uv run python -m unittest discover -s tests -v

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache data/output
