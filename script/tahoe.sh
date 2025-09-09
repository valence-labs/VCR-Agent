#!/bin/bash
#SBATCH --job-name=tahoe
#SBATCH --output=sbatch_log/%x.out
#SBATCH --error=sbatch_log/%x.err
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --wckey=hooke-predict
#SBATCH --mem=64GB

date

nvidia-smi

set -euo pipefail


# 1) Start embed server (8002)
( cd src/explain/literature/pubmedFastRAG && uv run embed.py --port 8002 --device cpu ) &
EMBED_PID=$!
until nc -z 127.0.0.1 8002; do sleep 1; done

# 2) Start Julia RAG server (8003)
( cd src/explain/literature/pubmedFastRAG && julia --project=. -t auto -e 'using Pkg; Pkg.instantiate(); include("rag.jl"); rag = RAGServer("../pubmed_vectors/pubmed_data.db"); start_server(rag)') &
JULIA_PID=$!
until nc -z 127.0.0.1 8003; do sleep 1; done

uv run src/explain/llm/generate.py \
    --experiment_name multi_tool \
    --wandb_mode online \
    --mode report-explain \
    --tool_list '["pubmed-fast-ner", "kg-ner-subgraph", "harmonizome", "wikipedia"]' \
    --folder_name tahoe \
    --kg_with_rel \
    --pert_path /rxrx/data/user/yunhui.jang/outgoing/hunflair/perturbation_ner_mapping_tahoe.json

# optional: stop servers after test
kill $EMBED_PID $JULIA_PID || true