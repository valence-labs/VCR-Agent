[![scorecard-score](https://github.com/recursionpharma/octo-guard-badges/blob/trunk/badges/repo/hooke-explain/maturity_score.svg?raw=true)](https://infosec-docs.prod.rxrx.io/octoguard/scorecards/hooke-explain)
[![scorecard-status](https://github.com/recursionpharma/octo-guard-badges/blob/trunk/badges/repo/hooke-explain/scorecard_status.svg?raw=true)](https://infosec-docs.prod.rxrx.io/octoguard/scorecards/hooke-explain)

# Hooke Explain

The codebase for the explain component of Hooke

## Quick Start

For detailed installation instructions, see [docs/installation.md](docs/installation.md).

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Basic Setup

**Automated Setup (Recommended)**
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:recursionpharma/hooke-explain.git
cd hooke-explain

# Run automated setup
make setup
# Verify installation
uv run python -c "import explain; print('OK')"
```

## Project Structure

- **src/explain/** - Main explanation package
- **cb-reach/** - Git submodule for CB Reach functionality
- **notebooks/** - Jupyter notebooks for experimentation
- **docs/** - Documentation

## Development

### Code Quality
```bash
# Using uv directly
uv run ruff check .
uv run ruff format .

# Using Makefile (recommended)
make lint
make format
make precommit  # runs both lint and format
```

### Makefile Commands
```bash
make help           # Show all available commands
make setup          # Complete project setup
make cbreach        # setup cb-reach submodule and dependencies
make test           # Run tests
```

### Working with cb-reach
Both the main project and cb-reach submodule use uv for dependency management. uv automatically handles virtual environments for each project.

## Documentation

- [Installation Guide](docs/installation.md) - Comprehensive setup instructions
- [CB-Reach README](cb-reach/README.md) - Submodule documentation
