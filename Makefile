.PHONY: help install-uv install setup-submodules cb-reach setup lint format test precommit digest

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "%-20s %s\n", $$1, $$2}'

install-uv: ## Install uv
	curl -LsSf https://python.prod.rxrx.io/rxrx-setup-uv.sh | sh

setup-submodules: ## Initialize and update git submodules
	git submodule update --init --recursive

install-poetry: ## Install Poetry via uv
	uv tool install poetry

cb-reach: setup-submodules install-poetry ## Update cb-reach submodule and install dependencies
	git submodule update --remote cb-reach
	cd cb-reach/enhanced-chat-client && uv tool run poetry install
	uv add --editable ./cb-reach/enhanced-chat-client

install: ## Install main project dependencies
	uv sync
	uv pip install -e .

setup: setup-submodules cb-reach install ## Complete project setup (submodules + all dependencies)
	@echo "✅ Setup complete! Run 'make help' to see available commands."

lint: ## Run linting with ruff
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

precommit: lint format ## Run pre-commit checks (lint and format)

digest: ## Generate a text digest of the entire codebase for use with Gemini
	gitingest . --exclude-pattern "uv.lock wandb/ outputs/ **/__pycache__ data/ cb-reach/"