import asyncio
import time
from typing import Any

from litellm import aembedding as litellm_aembedding
from loguru import logger

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
from explain.literature._prompt import COMPREHENSIVE_PROMPT, EVIDENCE_ONLY_PROMPT
from explain.llm import create_client


async def generate_evidence_based_answer(
    query: str,
    evidence: list[dict],
    model: str = "gpt-5",
    evidence_only: bool = True,
) -> str:
    """Generate a scientifically-grounded answer supported by literature evidence.

    Takes evidence passages from literature search and synthesizes them into a clear,
    well-founded answer to the scientific question. The answer quality depends on
    the relevance and comprehensiveness of the provided evidence.

    Args:
        query: Scientific question to answer
        evidence: List of evidence passages from vector search results with 'text' field
        model: LLM model to use for answer synthesis
        evidence_only: If True, answer strictly from evidence. If False, can supplement with knowledge.

    Returns:
        Evidence-supported answer to the scientific question
    """
    try:
        llm = create_client(provider="litellm", model=model)

        # Format evidence from vector search results with better structure
        evidence_texts = []
        for i, result in enumerate(evidence, 1):  # Start numbering from 1
            paper_id = result.get("paper_id")
            if paper_id:
                pmid_numeric = paper_id.replace("pubmed_", "").replace("PMC", "").replace("pmc_", "")
                source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_numeric}/" if pmid_numeric else "N/A"
            else:
                source_url = "Unknown"

            similarity = result.get("similarity_score", 0.0)
            text = result["text"]

            # Add metadata for context
            evidence_texts.append(f"Evidence {i} (Paper: {source_url}, Similarity: {similarity:.3f}):\n{text}")

        evidence_combined = "\n\n".join(evidence_texts)

        # Create different prompts based on answer mode
        if evidence_only:
            prompt = EVIDENCE_ONLY_PROMPT.format(query=query, evidence_combined=evidence_combined)
        else:
            prompt = COMPREHENSIVE_PROMPT.format(query=query, evidence_combined=evidence_combined)

        response = await llm.agenerate(messages=[{"role": "user", "content": prompt}])
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error generating evidence-based answer: {e}")
        return None


async def process_literature_query(
    query: str,
    keywords: list[str],
    top_k: int = 50,
    vector_top_k: int = 15,
    similarity_threshold: float | None = None,
    indexes: list[str] | None = None,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-5",
    generate_answer: bool = True,
    evidence_only: bool = True,
    build_search_index: bool = True,
    es_primitive_catalog: str | None = None,
    es_kwargs: dict[str, Any] = {},
) -> dict[str, Any]:
    """
    Generate evidence-based answers to scientific questions using literature search.

    This function implements the complete pipeline for answering scientific questions:
    1. Search literature using keywords (Elasticsearch)
    2. Process and embed new papers not yet in the knowledge base
    3. Find semantically relevant evidence using vector search
    4. Generate well-founded answers supported by the retrieved evidence

    The resulting answers are grounded in published scientific literature and can be
    configured to use only the evidence found or to supplement with scientific knowledge.

    Args:
        query: Scientific question to answer
        keywords: Keywords to search for relevant literature
        top_k: Maximum papers to retrieve from literature search
        vector_top_k: Maximum evidence passages to use for answer generation
        similarity_threshold: Minimum similarity score for evidence passages
        indexes: Literature databases to search (defaults to ['full'])
        embedding_model: Model for semantic similarity search
        llm_model: Model for generating evidence-based answers
        generate_answer: Whether to generate an answer (if False, returns only search results)
        evidence_only: If True, answer only from evidence. If False, can use scientific knowledge.
        build_search_index: Whether to build the search index in BigQuery
        es_primitive_catalog: Primitive catalog to use for the Elasticsearch search
        es_kwargs: Additional keyword arguments to pass to the Elasticsearch search

    Returns:
        Dictionary containing search results, evidence, and generated answer (if requested)
    """
    start_time = time.time()

    if indexes is None:
        indexes = ["full"]

    # Step 1: Initial keyword search using Elasticsearch
    logger.debug("Starting Elasticsearch search...")
    es_keywords = list(set(keywords))

    articles = await search_indexes(
        keywords=es_keywords, indexes=indexes, top_k=top_k, primitive_catalog=es_primitive_catalog, **es_kwargs
    )

    real_articles = [a for a in articles if a.get("id") != "fallback_id"]
    logger.debug(f"Elasticsearch returned {len(real_articles)} articles")

    # Step 2: Set up BigQuery table reference
    table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_NAME}"

    # Step 3: Check which papers need processing
    paper_ids = [article["id"] for article in real_articles]
    existing_paper_ids = await check_existing_papers_in_bigquery(paper_ids, table_ref)
    papers_to_process = [a for a in real_articles if a["id"] not in existing_paper_ids]

    logger.debug(f"Need to process {len(papers_to_process)} new papers")

    # Step 4: Process new papers (chunk, embed, upload) - PARALLEL PROCESSING
    all_chunks: list[dict] = []
    if papers_to_process:
        # Process papers in parallel for better performance
        async def process_paper(article):
            paper_content = (
                f"Title: {article.get('title', '')}\n"
                f"Abstract: {article.get('abstract', '')}\n"
                f"Content: {article.get('content', '')}"
            )
            return chunk_papers(paper_content=paper_content, paper_id=article["id"])

        # Run chunking in parallel
        chunk_tasks = [process_paper(article) for article in papers_to_process]
        chunk_results = await asyncio.gather(*chunk_tasks)

        # Flatten results
        for chunks in chunk_results:
            all_chunks.extend(chunks)

        logger.debug(f"Created {len(all_chunks)} chunks from {len(papers_to_process)} papers in parallel")

        # Embed chunks using litellm via shared util (async batched)
        embedded_chunks = await aembed_chunks(chunks=all_chunks, embedding_model=embedding_model)

        # Upload to BigQuery using concurrent batch insert
        await aload_to_bigquery(
            embedded_chunks=embedded_chunks,
            table_name=BIGQUERY_TABLE_NAME,
            project_id=BIGQUERY_PROJECT_ID,
            dataset_id=BIGQUERY_DATASET_ID,
            use_gcs=True,
            build_index=build_search_index,
        )

    # Step 5: Perform vector search
    logger.debug("Starting vector search...")

    embedding_response = await litellm_aembedding(model=embedding_model, input=query)
    query_embedding = embedding_response.data[0]["embedding"]

    # Perform vector search
    vector_results = await perform_vector_search(
        query_embedding=query_embedding,
        paper_ids=paper_ids,
        table_ref=table_ref,
        top_k=vector_top_k,
        similarity_threshold=similarity_threshold,  # Filter low-quality results for answers
    )

    # Step 6: Generate evidence-based answer
    generated_answer = None
    if generate_answer and vector_results:
        logger.debug(
            f"Generating evidence-based answer using mode: {'evidence_only' if evidence_only else 'comprehensive'}"
        )
        generated_answer = await generate_evidence_based_answer(
            query=query,
            evidence=vector_results,  # Only use vector results as evidence
            model=llm_model,
            evidence_only=evidence_only,
        )

    processing_time = time.time() - start_time

    # Combine results
    combined_results = {"elasticsearch_articles": real_articles, "vector_search_results": vector_results}

    response = {
        "query": query,
        "keywords": es_keywords,
        "num_elasticsearch_results": len(real_articles),
        "num_vector_results": len(vector_results),
        "processing_time": processing_time,
        "evidence_only": evidence_only,
        "results": combined_results,
    }

    if generated_answer:
        response["generated_answer"] = generated_answer

    return response
