# Data Generation Workflow

An LLM data generation agent that integrates multiple tools (NER, PubMedSearch, KG search, Wikipedia, Harmonizome) to generate structured biological explanations. Includes an evaluation framework with token-based, embedding-based, and LLM-judging metrics.

## Workflow Overview

1. **Input**: Perturbation + Context data (JSON)
2. **Preprocessing**: NER entity extraction (HunFlair2) + PubChem synonym search
3. **LLM Agent**: Report and structured explanation generation with external tools
4. **Evaluation**: Token-based, Embedding-based, LLM-judging

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Environment Variables

Create a `.env` file (or export) with:

```bash
# Required
DATA_DIR=data/curation_v1              # Path to directory with action_primitives.json, templates/, mondo.json

# Required for KG tool
STARK_PRIMEKG_DIR=/path/to/stark_prime_kg   # Directory with edge_index.pt, node_info.json, etc. (https://stark.stanford.edu/dataset_prime.html)
DRUGBANK_XML_PATH=/path/to/full_database.xml  # DrugBank XML (optional, for chemical entity matching)

# Required for LLM providers
ANTHROPIC_API_KEY=...                  # For Anthropic Claude
# or
OPENAI_API_KEY=...                     # For OpenAI
```

### 3. PubMed RAG Server (required for `pubmed-fast-ner` tool)

Clone repositories, download the PubMed database (~34GB fully built) and binary search data (~2.5GB). If the download process does not work well, refer to https://github.com/domluna/pubmedFastRAG.git for full explanation:

```bash
git clone https://github.com/kyunghyuncho/pubmed-vectors.git
git clone https://github.com/domluna/pubmedFastRAG.git

# Download PubMed SQLite database (takes a while)
python pubmed-vectors/download_pubmed.py

# Download binary RAG search data (from Google Drive, ~2.5GB)
pip install gdown
gdown "1LuCaUcILQuQgkDm3_tWBWr4X7AQ518kX" -O pubmedFastRAG/bindata.zip
cd pubmedFastRAG && unzip bindata.zip -d bindata/ && rm bindata.zip && cd ..

# Move database to data/
mv pubmed_data.db data/pubmed_data.db
```

Install Python dependencies for the embed server:

```bash
pip install fastapi uvicorn einops
```

Install Julia (if not already installed) and resolve Julia packages:

```bash
cd pubmedFastRAG
julia --project=. -e 'using Pkg; Pkg.update(); Pkg.precompile()'
cd ..
```

Unzip data/mondo.json.zip.

## Input Format

The input is a JSON list of perturbation records. Some fields may be missing.

```json
[
  {
    "index": 0,
    "perturbation": {
      "context": [
        {
          "perturbation type": "soluble factor",
          "description": "Soluble factor addition of VEGF",
          "cell_type": "N/A",
          "disease_model": "Angiogenic factor/tumors",
          "cell type": null,
          "subtype": null
        }
      ],
      "perturbations": [
        {
          "type": "chemical",
          "smiles": "CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O",
          "name": "Nintedanib",
          "target": "VEGFR",
          "moa_type": "antibody",
          "known targets": []
        }
      ]
    }
  }
]
```

## Preprocessing

Extracts NER entities and PubChem compound information from the input perturbations:

```bash
INPUT_PATH=data/example_input.json
PREPROCESSED_PATH=data/example_preprocessed.json

DATA_DIR=data/curation_v1 uv run script/preprocess.py --input_path "$INPUT_PATH" --preprocessed_path "$PREPROCESSED_PATH"
```

The preprocessed output adds `perturbation_entity` and `context_entity` fields (NER results) and optional `pubchem_info` for chemical perturbations.

## LLM Agent and Evaluation

The agent generates both a report and structured explanation using external tools.

### Available Tools

| Tool | Description | Requires |
|------|-------------|----------|
| `pubmed-fast-ner` | PubMed paper search via NER entities | Julia RAG server (ports 8002, 8003) |
| `kg-ner` | KG node lookup via NER entities | `STARK_PRIMEKG_DIR` env var |
| `harmonizome` | Gene/gene-set information | Internet access |
| `wikipedia` | Wikipedia articles for NER entities | Internet access |

### Running

```bash
# 1) Start PubMed servers (skip if not using pubmed-fast-ner)
( cd pubmedFastRAG && uv run embed.py --port 8002 --device cpu ) &
EMBED_PID=$!
until nc -z 127.0.0.1 8002; do sleep 1; done

( cd pubmedFastRAG && julia --project=. -t auto -e \
  'using Pkg; Pkg.instantiate(); include("rag.jl"); rag = RAGServer("../data/pubmed_data.db"); start_server(rag; port=8003)' ) &
JULIA_PID=$!
until nc -z 127.0.0.1 8003; do sleep 1; done

# 2) Run LLM agent
DATA_DIR=data/curation_v1 uv run script/generate.py \
  --experiment_name multi_tool_order \
  --wandb_mode disabled \
  --mode report-explain \
  --model_type litellm \
  --tool_list '["pubmed-fast-ner", "kg-ner", "harmonizome", "wikipedia"]' \
  --folder_name multi_tool \
  --pert_path "$PREPROCESSED_PATH" \
  --kg_with_rel

# 3) Stop servers (optional)
kill $EMBED_PID $JULIA_PID || true
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `litellm` | LLM provider (`litellm`, `anthropic`, `openai`, `gemini`). Use `litellm` with API keys in `.env` |
| `--tool_list` | `["kg-ner"]` | JSON list of tools to use |
| `--mode` | `report-explain` | `report-explain` or `explain-only` |
| `--kg_with_rel` | `false` | Include KG relation info |
| `--wandb_mode` | `disabled` | W&B logging (`online`, `offline`, `disabled`) |
| `--max_items` | `0` | Limit perturbations to process (0 = all) |
| `--pert_path` | `data/perturbation_ner_mapping.json` | Preprocessed perturbation file |

## Notebooks

- `notebooks/data-generation/data_generation.ipynb` - Interactive data generation walkthrough
- `notebooks/data-generation/kg_ner.ipynb` - KG NER integration demo

## Troubleshooting

### Ports already in use
```bash
lsof -iTCP:8002 -sTCP:LISTEN -n -P
lsof -iTCP:8003 -sTCP:LISTEN -n -P
lsof -tiTCP:8002 -sTCP:LISTEN | xargs -r kill
lsof -tiTCP:8003 -sTCP:LISTEN | xargs -r kill
```

### Julia RAG server not starting
Ensure Julia is installed and on PATH, and that `pubmed_data.db` exists under `data/`.

### KG data not found
Set `STARK_PRIMEKG_DIR` to the directory containing `edge_index.pt`, `node_info.json`, and other KG files. Set `DATA_DIR` to the directory containing `mondo.json` and `action_primitives.json`.
