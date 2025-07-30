# Installation Guide

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) - Package manager (used for everything)
- Git with SSH access to cb-reach repository

## Quick Setup

### 1. Clone with Submodules
```bash
git clone --recurse-submodules git@github.com:recursionpharma/hooke-explain.git
cd hooke-explain

# If already cloned without submodules:
git submodule update --init --recursive
```

### 3. Install cb-reach Submodule
```bash
cd cb-reach

# Convert Poetry project to uv and install
uv sync

# If uv sync fails, install dependencies manually
# uv pip install -e .

cd ..
```

### 3. Install Main Project (hooke-explain)
```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```


### 4. Verify Installation
```bash
# Test main project imports
uv run python -c "import explain; print('Main project OK')"

# Test cb-reach imports
cd cb-reach
uv run python -c "import cb_reach; print('cb-reach OK')"
cd ..
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
make setup-submodules    # Initialize git submodules
make install             # Install main project
make install-cbreach     # Install cb-reach dependencies
make install-all         # Install both main project and cb-reach
make setup               # Complete setup (submodules + dependencies)
make update-cbreach      # Update cb-reach and install dependencies
```

### Development Commands
```bash
make lint               # Run ruff linting
make format             # Format code with ruff
make precommit          # Run both lint and format
make test               # Run main project tests
make test-cbreach       # Run cb-reach tests
```

## Troubleshooting

### Common Issues

**Import errors from cb-reach:**
```bash
# Check submodule status
git submodule status

# Reinitialize if needed
git submodule update --init --recursive

# Install cb-reach dependencies with uv
cd cb-reach && uv sync && cd ..
```

**Python version / dependencies conflicts:**
- Both main project and cb-reach use Python 3.12
- Ensure you have Python 3.12 installed
- cb-reach originally uses Poetry but we convert to uv

### Environment Management

**uv automatically manages virtual environments:**
```bash
# Main project
cd /path/to/hooke-explain
uv run python  # automatically uses project's virtual environment

# cb-reach
cd cb-reach
uv run python  # automatically uses cb-reach's virtual environment
```

**Manual environment activation (if needed):**
```bash
# Main project
source .venv/bin/activate
``` 