# Installation Guide

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) - Package manager (used for everything)
- Git access for cloning the repository

## Quick Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd 
```

### 2. Install Main Project ()
```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### 3. Verify Installation
```bash
# Test main project imports
uv run python -c "import explain; print('Main project OK')"

# Test project structure
uv run python -c "from explain.eval.tools import knowledge_graph; print('Tools OK')"
```

## Development Workflow

### Main Project Commands
```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .

# Add dependencies
uv add <package-name>
uv add --dev <dev-package-name>

# Update dependencies
uv sync --upgrade
```

## Using Makefile Commands

The project includes convenient Makefile commands for common tasks:

### Setup Commands
```bash
make help                # Show all available commands
make install-uv          # Install uv package manager

make install             # Install main project
make install             # Install project dependencies
make setup               # Complete setup (install dependencies)
```

### Development Commands
```bash
make lint               # Run ruff linting
make format             # Format code with ruff
make precommit          # Run both lint and format
make test               # Run tests
```

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Reinstall project dependencies
make install

# Verify installation
uv run python -c "import explain; print('Installation OK')"
```

**Python version / dependencies conflicts:**
- Project requires Python 3.12
- Ensure you have Python 3.12 installed


**Manual environment activation (if needed):**
```bash
# Main project
source .venv/bin/activate
``` 