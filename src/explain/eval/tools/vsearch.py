import asyncio
import time
from typing import Any

from litellm import aembedding as litellm_aembedding
from loguru import logger
from pydantic import BaseModel, Field

from explain.literature._bq_process_paper import (
    BIGQUERY_DATASET_ID,
    BIGQUERY_PROJECT_ID,
    BIGQUERY_TABLE_NAME,
    aembed_chunks,
    aload_to_bigquery,
    check_existing_papers_in_bigquery,
    chunk_papers,
    perform_vector_search,
)
from explain.literature._esearch_utils import search_indexes
from explain.llm import create_client

from ._base import ToolVerifier


class VectorSearchArgs(BaseModel):
    query: str = Field(..., description="The main question or query to search for")
    keywords: list[str] = Field(..., description="List of keywords to enhance the search")
    top_k: int = Field(50, ge=1, le=200, description="Maximum number of documents to retrieve from elasticsearch")
    vector_top_k: int = Field(15, ge=1, le=50, description="Maximum number of vector search results")
    indexes: list[str] | None = Field(
        default=["full"],
        description="List of indexes to search. Available: 'full', 'fda', 'pdf', 'informa', 'informa-drugs'",
    )
    use_hyde: bool = Field(
        default=False, description="Whether to use hypothetical document embedding for enhanced retrieval"
    )


async def generate_hypothetical_document(query: str, model: str = "gpt-5") -> str:
    """Generate a hypothetical document for HyDE (Hypothetical Document Embeddings)."""
    try:
        llm = create_client(provider="litellm", model=model)

        prompt = f"""
        You are an expert document creator. Given a question, generate a detailed document that would directly answer this question.
        The document should be approximately 2000 characters long and provide an in-depth, informative answer to the question.
        Write as if this document is from an authoritative source on the subject. Include specific details, facts, and explanations.
        Do not mention that this is a hypothetical document - just write the content directly.
        Use scientific terminology and references where appropriate. Include relevant acronyms that might appear in scientific literature.
        
        Question: {query}
        """

        response = await llm.agenerate(messages=[{"role": "user", "content": prompt}])
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error generating hypothetical document: {e}")
        return query  # Fallback to original query


class VectorSearchVerifier(ToolVerifier):
    """
    Enhanced literature search tool that combines Elasticsearch retrieval with BigQuery vector search.

    This tool performs a two-stage search:
    1. Initial keyword-based search using Elasticsearch
    2. Vector similarity search using embeddings stored in BigQuery

    The tool also handles paper chunking, embedding, and storage for new papers.
    """

    name = "literature_vector_search"
    description = (
        "Enhanced literature search combining keyword search with vector similarity search. "
        "Performs initial Elasticsearch search, then uses BigQuery vector search for semantic similarity. "
        "Automatically chunks and embeds new papers for future searches."
    )
    args_schema = VectorSearchArgs

    def _tool_logic(self, args: VectorSearchArgs) -> tuple[float, dict[str, Any]]:
        try:
            return asyncio.run(self._async_tool_logic(args))
        except Exception as e:
            logger.error(f"Error in vector search tool: {e}")
            return 0.0, {"error": f"Vector search failed: {e}"}

    async def _async_tool_logic(self, args: VectorSearchArgs) -> tuple[float, dict[str, Any]]:
        start_time = time.time()

        # Step 1: Initial keyword search using existing Elasticsearch functionality
        logger.debug("Starting Elasticsearch search...")
        # Keep provided keywords; optionally augment with simple NER later
        es_keywords = list(set(args.keywords))

        articles = await search_indexes(keywords=es_keywords, indexes=args.indexes, top_k=args.top_k)

        real_articles = [a for a in articles if a.get("id") != "fallback_id"]
        logger.debug(f"Elasticsearch returned {len(real_articles)} articles")

        if not real_articles:
            return 0.0, {
                "query": args.query,
                "keywords": es_keywords,
                "elasticsearch_results": 0,
                "vector_results": 0,
                "results": [],
                "processing_time": time.time() - start_time,
            }

        # Step 2: Set up BigQuery and embedding model
        table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_NAME}"

        # Step 3: Check which papers need processing
        paper_ids = [article["id"] for article in real_articles]
        existing_paper_ids = await check_existing_papers_in_bigquery(paper_ids, table_ref)
        papers_to_process = [a for a in real_articles if a["id"] not in existing_paper_ids]

        logger.debug(f"Need to process {len(papers_to_process)} new papers")

        # Step 4: Process new papers (chunk, embed, upload)
        if papers_to_process:
            all_chunks: list[dict] = []
            for article in papers_to_process:
                paper_content = (
                    f"Title: {article.get('title', '')}\n"
                    f"Abstract: {article.get('abstract', '')}\n"
                    f"Content: {article.get('content', '')}"
                )
                chunks = chunk_papers(paper_content=paper_content, paper_id=article["id"])  # token-aware
                all_chunks.extend(chunks)

            logger.debug(f"Created {len(all_chunks)} chunks")

            # Embed chunks using litellm via shared util (async batched)
            embedded_chunks = await aembed_chunks(chunks=all_chunks)

            # Upload to BigQuery using concurrent batch insert
            await aload_to_bigquery(
                embedded_chunks=embedded_chunks,
                table_name=BIGQUERY_TABLE_NAME,
                project_id=BIGQUERY_PROJECT_ID,
                dataset_id=BIGQUERY_DATASET_ID,
            )

        # Step 5: Perform vector search
        logger.debug("Starting vector search...")

        # Generate query embedding (with optional HyDE)
        search_query = args.query
        if args.use_hyde:
            hypothetical_doc = await generate_hypothetical_document(args.query)
            search_query = hypothetical_doc
            logger.debug("Using HyDE document for vector search")

        # Get query embedding via litellm for consistency
        query_embedding_list = await litellm_aembedding(model="text-embedding-3-small", input=search_query)
        # litellm returns a list when input is a string as well; use the first element if nested
        query_embedding = (
            query_embedding_list[0]
            if query_embedding_list and isinstance(query_embedding_list[0], list)
            else query_embedding_list
        )

        # Perform vector search
        vector_results = await perform_vector_search(
            query_embedding=query_embedding, paper_ids=paper_ids, table_ref=table_ref, top_k=args.vector_top_k
        )

        # Step 6: Calculate reward and prepare response
        total_results = len(real_articles) + len(vector_results)
        reward = 1.0 if total_results > 0 else 0.0

        processing_time = time.time() - start_time

        # Combine results
        combined_results = {"elasticsearch_articles": real_articles, "vector_search_results": vector_results}

        return reward, {
            "query": args.query,
            "keywords": es_keywords,
            "elasticsearch_results": len(real_articles),
            "vector_results": len(vector_results),
            "papers_processed": len(papers_to_process),
            "total_chunks_created": len(all_chunks) if papers_to_process else 0,
            "processing_time": processing_time,
            "bigquery_table": BIGQUERY_TABLE_NAME,
            "used_hyde": args.use_hyde,
            "results": combined_results,
        }
