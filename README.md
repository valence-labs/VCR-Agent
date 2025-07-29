# hooke-explain
The codebase for the explain component of Hooke



## Installation:

```bash
#Install uv:
make install-uv

# Create a Py3.12 env:
uv venv --python 3.12
source .venv/bin/activate

#Install dependencies:
make install
```


## Development:
To modify dependencies:
```bash
# To add a new dependency:
uv add torch

# To remove a dependency:
uv remove torch
```