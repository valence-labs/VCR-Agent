.PHONY: help install-uv install setup lint format test precommit digest

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "%-20s %s\n", $$1, $$2}'

install-uv: ## Install uv
	curl -LsSf https://python.prod.rxrx.io/rxrx-setup-uv.sh | sh

install-poetry: ## Install Poetry via uv
	uv tool install poetry

install: ## Install main project dependencies
	uv sync
	uv pip install -e .

setup: install ## Complete project setup (all dependencies)
	@echo "✅ Setup complete! Run 'make help' to see available commands."

lint: ## Run linting with ruff
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

precommit: lint format ## Run pre-commit checks (lint and format)

digest: ## Generate a text digest of the entire codebase for use with Gemini
	gitingest . --exclude-pattern "uv.lock wandb/ outputs/ **/__pycache__ data/"