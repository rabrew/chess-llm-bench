PYTHON ?= venv/bin/python

.PHONY: check test lint
check: lint test

lint:
	$(PYTHON) -m ruff check --no-cache src scripts tests dashboard

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest tests -q -p no:cacheprovider
