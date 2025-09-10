#!/bin/bash
#SBATCH --job-name=multi_tool
#SBATCH --output=sbatch_log/%x.out
#SBATCH --error=sbatch_log/%x.err
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --wckey=hooke-predict

date

nvidia-smi

set -euo pipefail


# 1) Start embed server (8002)
( cd pubmedFastRAG && uv run embed.py --port 8002 --device cpu ) &
EMBED_PID=$!
until nc -z 127.0.0.1 8002; do sleep 1; done

# 2) Start Julia RAG server (8003)
( cd pubmedFastRAG && julia --project=. -t auto -e 'using Pkg; Pkg.instantiate(); include("rag.jl"); rag = RAGServer("../data/pubmed_data.db"); start_server(rag)') &
JULIA_PID=$!
until nc -z 127.0.0.1 8003; do sleep 1; done

uv run src/explain/llm/generate.py \
    --experiment_name multi_tool_order \
    --wandb_mode online \
    --mode report-explain \
    --tool_list '["pubmed-fast-ner", "kg-ner", "harmonizome", "wikipedia"]' \
    --folder_name multi_tool ;

# optional: stop servers after test
kill $EMBED_PID $JULIA_PID || true