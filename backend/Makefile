.PHONY: dev test format lint

dev:
	uvicorn src.agent.app:app --reload --port 8000

test:
	python -m pytest tests/ -v

format:
	ruff format src/ examples/
	ruff check --fix src/ examples/

lint:
	ruff check src/ examples/
