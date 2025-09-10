( cd pubmedFastRAG && uv run embed.py --port 8002 --device cpu ) &
EMBED_PID=$!
until nc -z 127.0.0.1 8002; do sleep 1; done

# 2) Start Julia RAG server (8003)
( cd pubmedFastRAG && julia --project=. -t auto -e 'using Pkg; Pkg.instantiate(); include("rag.jl"); rag = RAGServer("../data/pubmed_data.db"); start_server(rag; port=8003)') &
JULIA_PID=$!
until nc -z 127.0.0.1 8003; do sleep 1; done