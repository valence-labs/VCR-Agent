.PHONY: help install-uv install

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "%-15s %s\n", $$1, $$2}'

install-uv: ## Install uv
	curl -LsSf https://python.prod.rxrx.io/rxrx-setup-uv.sh | sh

install: ## Install dependencies
	uv pip install -e .

digest: ## Generate a text digest of the entire codebase for use with Gemini
	gitingest . --exclude-pattern "uv.lock wandb/ outputs/ **/__pycache__"