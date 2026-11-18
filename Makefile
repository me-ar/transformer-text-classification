.PHONY: install test lint benchmark smoke
install:
	pip install -r requirements-dev.txt && pip install -e .
test:
	pytest -q
lint:
	ruff check src tests
benchmark:
	python -m textclf.experiment
smoke:
	python -m textclf.experiment --smoke
