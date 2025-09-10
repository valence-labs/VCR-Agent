# Data generation workflow

This introduces the data generation agent LLM that integrates multiple tools including NER, PubMedSearch, KG search, Wikipedia, and Harmonize. Additionally, it includes the evaluation framework that compares the generated and GT explanation with token-based and embedding-based metrics.

## Workflow overview
The data generation process follows:
1. Input: Perturbation + Context data
2. Preprocessing (NER, synonym search): Preprocess the perturbation data for faster data generation
    - NER: To find the related entities (chemical, disease, gene) with hunflair2
    - Synonym search: To find the synonyms of chemicals with PubChem
3. LLM agent with tools: PubMed, StarkPrimeKG, Harmonizome, Wikipedia
4. Evaluation: Token-based, Embedding-based, LLM-judging

## Setting


Clone the PubMed-related repositories and download the pubmed_data.db file to data/
```bash
git clone https://github.com/kyunghyuncho/pubmed-vectors.git
git clone https://github.com/domluna/pubmedFastRAG/tree/main
```

```bash
python pubmed-vectors/download_pubmed.py
```

If Julia is not installed, install Julia

```bash
cd pubmedFastRAG
JULIA_VERSION=1.11.6
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-$JULIA_VERSION-linux-x86_64.tar.gz
tar -xvzf julia-$JULIA_VERSION-linux-x86_64.tar.gz
ln -s julia-$JULIA_VERSION julia

export PATH="julia/bin:$PATH" # add to PATH (put this in ~/.bashrc or your sbatch script. Modify this to the corresponding Julia path)
julia --version
```

## Input
The input format follows the example. Note that some entities could be missing.

```json
[{  'index': 0,
    "perturbation": {
        'context': [{
            'perturbation type': 'soluble factor', 
            'description': 'Soluble factor addition of VEGF', 'cell_type': 'N/A', 
            'disease_model': 'Angiogenic factor/tumors'
            'cell type': ,
            'subtype': ,
        }], 
        'perturbations': [{
            'type': 'chemical', 
            'smiles': "CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O", 
            'name': 'Nintedanib', 
            'target': 'VEGFR', 
            'moa_type': 'antibody'
            'known targets': ,
        }]
    }
},
...
]
```

## Preprocessing

```bash
# input perturbation path
INPUT_PATH=path/to/perturbations.json
# processed perturbation path (output)
PREPROCESSED_PATH=path/to/preprocessed.json

uv run script/preprocess.py --input_path "$INPUT_PATH" --preprocessed_path "$PREPROCESSED_PATH"
```

The preprocessed format follows:

```json
[{  'index': 0,
    "perturbation": {
            "context": {
                "perturbation_type": "soluble factor",
                "description": "Soluble factor addition of VEGF",
                "cell_type": "N/A",
                "disease_model": "Angiogenic factor/tumors"
            },
            "perturbations": [{
                "type": "chemical",
                "smiles": "CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O",
                "name": "Nintedanib",
                "target": "VEGFR",
                "moa_type": "inhibitor",
                # added from synonym search
                "pubchem_info": {
                    "cid": 135423438,
                    "name": "methyl 2-hydroxy-3-[N-[4-[methyl-[2-(4-methylpiperazin-1-yl)acetyl]amino]phenyl]-C-phenylcarbonimidoyl]-1H-indole-6-carboxylate",
                    "smiles": "CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O",
                    "chembl_id": "CHEMBL2136735",
                    "inchikey": "CPMDPSXJELVGJG-UHFFFAOYSA-N",
                    "synonyms": [
                        "Nintedanib",
                        "656247-17-5",
                        "Vargatef",
                        ...
                    ]
                }
            }]
        },
        # added from NER
        "perturbation_entity": {
            "Nintedanib": "Chemical"
        },
        "context_entity": {
            "VEGF": "Gene"
        }
},
...
]
```

## LLM agent & Evaluation

LLM agent generates both report and structured explanation incorporating the knowledge from external tools.

Four choices of tools: "pubmed-fast-ner", "kg-ner", "harmonizome", "wikipedia"
- pubmed-fast-ner: Find the related papers from PubMed with the list of NER entities
- kg-ner: Find the related nodes of NER entities and its 1-hop neighborhoods from StakrPrimeKG
- harmonizome: Get the gene-related information
- wikipedia: Get the related wikipedia documents of NER entities

Note that one need to run Julia RAG server for pubmed-fast-ner tool.

```bash
# 1) Start embed server (8002) (Can be omitted when not using pubmed-fast-ner tool)
( cd pubmedFastRAG && uv run embed.py --port 8002 --device cpu ) &
EMBED_PID=$!
until nc -z 127.0.0.1 8002; do sleep 1; done

# 2) Start Julia RAG server (8003) (Can be omitted when not using pubmed-fast-ner tool)
( cd pubmedFastRAG && julia --project=. -t auto -e 'using Pkg; Pkg.instantiate(); include("rag.jl"); rag = RAGServer("../data/pubmed_data.db"); start_server(rag; port=8003)' ) &
JULIA_PID=$!
until nc -z 127.0.0.1 8003; do sleep 1; done

# 3) Run LLM agent
uv run script/generate.py \
  --experiment_name multi_tool_order \
  --wandb_mode online \
  --mode report-explain \
  --tool_list '["pubmed-fast-ner", "kg-ner", "harmonizome", "wikipedia"]' \
  --folder_name multi_tool \
  --pert_path "$PREPROCESSED_PATH" \
  --kg_with_rel

# optional: stop servers after test
kill $EMBED_PID $JULIA_PID || true
```

## Troubleshooting 

### ports (address already in use)
```bash
lsof -iTCP:8002 -sTCP:LISTEN -n -P
lsof -iTCP:8003 -sTCP:LISTEN -n -P
lsof -tiTCP:8002 -sTCP:LISTEN | xargs -r kill
lsof -tiTCP:8003 -sTCP:LISTEN | xargs -r kill
```