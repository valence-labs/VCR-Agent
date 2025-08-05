[![scorecard-score](https://github.com/recursionpharma/octo-guard-badges/blob/trunk/badges/repo/hooke-explain/maturity_score.svg?raw=true)](https://infosec-docs.prod.rxrx.io/octoguard/scorecards/hooke-explain)
[![scorecard-status](https://github.com/recursionpharma/octo-guard-badges/blob/trunk/badges/repo/hooke-explain/scorecard_status.svg?raw=true)](https://infosec-docs.prod.rxrx.io/octoguard/scorecards/hooke-explain)

The codebase for the explain component of Hooke, a framework for generating, verifying, and evaluating scientific explanations using LLMs.

## Quick Links

- **[Documentation](./docs/)**: Detailed information on installation, architecture, and usage.
- **[Notebooks](./notebooks/)**: Examples and experiments.

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
make lint
make format
make precommit  # runs both lint and format
```

You can also use ruff command directly if you prefer (e.g: `uv run ruff check --fix`)

### Makefile Commands
```bash
make help           # Show all available commands
make setup          # Complete project setup
make cb-reach        # setup cb-reach submodule and dependencies
make test           # Run tests
```

### Working with cb-reach
You might need to install the cb-reach repo to get access to the programmatic API of enhanced-chat. In our setup we do not want to alter the original code, so you are stuck with using `poetry` for installation. <>
## Documentation

- [Installation Guide](docs/installation.md) - Comprehensive setup instructions
