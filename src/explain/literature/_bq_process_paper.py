import asyncio
import os
from math import ceil

import dotenv
import tiktoken
from google.cloud import bigquery
from litellm import aembedding as litellm_aembedding
from loguru import logger

dotenv.load_dotenv()

BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "rxrx-medchem-auto-dev")
BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "hooke-explain-dev")
BIGQUERY_TABLE_NAME = os.getenv("BIGQUERY_TABLE_NAME", "elastic_search_index")


def chunk_text(text: str, chunk_size: int, overlap_size: int, dockey: str) -> list[dict]:
    """Splits text into chunks with overlap.

    Args:
        text: The text to chunk
        chunk_size: The size of each chunk
        overlap_size: The size of the overlap
        dockey: The key of the document

    Returns:
        A list of dictionaries, each representing a chunk with text, metadata, and IDs
    """
    try:
        content = text

        enc = tiktoken.get_encoding("cl100k_base")
        content_tokens = enc.encode(content)

        if not content_tokens:
            return []

        char_count = len(content)
        token_count = len(content_tokens)
        chars_per_token = char_count / token_count
        chunk_tokens = chunk_size / chars_per_token
        overlap_tokens = overlap_size / chars_per_token
        chunk_count = ceil(token_count / chunk_tokens)

        chunks = []
        for i in range(chunk_count):
            start_idx = max(int(i * chunk_tokens - overlap_tokens), 0)
            end_idx = int((i + 1) * chunk_tokens + overlap_tokens)
            split_part = content_tokens[start_idx:end_idx]
            chunk_text = enc.decode(split_part)

            chunk = {
                "text": chunk_text,
                "chunk_index": i + 1,
                "total_chunks": chunk_count,
                "chunk_id": f"{dockey}_{i + 1}",
            }
            chunks.append(chunk)

        return chunks
    except Exception as e:
        logger.error(f"Error chunking text for {dockey}: {e}")
        return []


def chunk_papers(paper_content: str, paper_id: str) -> list:
    """Chunk the paper content into smaller chunks.

    Args:
        paper_content: The content of the paper to chunk
        paper_id: Unique identifier for the paper

    Returns:
        List of dictionaries, each representing a chunk with text, metadata, and IDs
    """
    chunk_size = 7000
    overlap_size = 250

    chunks = chunk_text(
        text=paper_content,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        dockey=paper_id,
    )

    for chunk in chunks:
        chunk["paper_id"] = paper_id

    return chunks


async def aembed_chunks(chunks: list, embedding_model: str = "text-embedding-3-small") -> list:
    """Embed the chunks using an embedding model.

    Args:
        chunks: List of chunk dictionaries with text content

    Returns:
        List of dictionaries containing chunks with their embeddings added
    """
    try:
        embedded_chunks = []
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            embeddings = await litellm_aembedding(model=embedding_model, input=texts)

            for j, chunk in enumerate(batch):
                embedded_chunk = chunk.copy()
                embedded_chunk["embedding"] = embeddings[j]
                embedded_chunks.append(embedded_chunk)
        return embedded_chunks
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        return chunks


async def aload_to_bigquery(
    embedded_chunks: list,
    table_name: str = BIGQUERY_TABLE_NAME,
    project_id: str = BIGQUERY_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
):
    """Upload embedded chunks directly to BigQuery without using GCS asynchronously.

    Args:
        embedded_chunks: List of dictionaries containing chunks with their embeddings
        table_name: Name of the table to load data into

    Returns:
        None
    """
    bq_client = bigquery.Client(project=project_id)
    table_ref_string = f"{project_id}.{dataset_id}.{table_name}"

    # Schema definition based on the embedded_chunks structure
    schema = [
        bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("total_chunks", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("paper_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]

    try:
        dataset_ref = f"{project_id}.{dataset_id}"

        # Check if dataset exists
        try:
            await asyncio.to_thread(bq_client.get_dataset, dataset_ref)
            dataset_exists = True
        except Exception:
            dataset_exists = False

        if not dataset_exists:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            await asyncio.to_thread(bq_client.create_dataset, dataset)
            logger.info(f"Created dataset {dataset_ref}")

        # Check if table exists
        try:
            await asyncio.to_thread(bq_client.get_table, table_ref_string)
            table_exists = True
        except Exception:
            table_exists = False

        if not table_exists:
            table = bigquery.Table(table_ref_string, schema=schema)
            await asyncio.to_thread(bq_client.create_table, table)
            logger.info(f"Created table {table.table_id}")

        table = await asyncio.to_thread(bq_client.get_table, table_ref_string)

        rows_to_insert = []
        for chunk in embedded_chunks:
            if "embedding" not in chunk:
                logger.debug(f"Skipping chunk {chunk.get('chunk_id', 'unknown')} without embedding")
                continue

            row = {
                "text": chunk["text"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "paper_id": chunk["paper_id"],
                "embedding": chunk["embedding"],
            }
            rows_to_insert.append(row)

        if not rows_to_insert:
            logger.debug("No valid chunks to insert into BigQuery")
            return

        # Process batches concurrently
        batch_size = 100
        tasks = []

        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i : i + batch_size]
            task = asyncio.create_task(insert_batch(bq_client, table, batch, i))
            tasks.append(task)

        # Wait for all insert operations to complete
        await asyncio.gather(*tasks)

        # Verify data was inserted correctly
        count_query = f"""
        SELECT COUNT(*) as row_count 
        FROM `{table_ref_string}`
        """  # noqa

        query_job = await asyncio.to_thread(bq_client.query, count_query)
        results = await asyncio.to_thread(list, query_job.result())
        if results:
            row_count = results[0].row_count
            logger.info(f"Table now contains {row_count} rows")

        logger.info(f"Successfully loaded chunks to BigQuery table {table_ref_string}")

    except Exception as e:
        logger.error(f"Error in BigQuery load process: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


async def insert_batch(client, table, batch, batch_index):
    """Insert a batch of rows into BigQuery asynchronously."""
    try:
        errors = await asyncio.to_thread(client.insert_rows_json, table, batch)

        if errors:
            logger.error(f"Errors inserting batch starting at index {batch_index}: {errors}")
        else:
            logger.info(f"Inserted batch of {len(batch)} rows starting at index {batch_index}")

        return errors
    except Exception as e:
        logger.error(f"Exception inserting batch at index {batch_index}: {e}")
        raise


async def check_existing_papers_in_bigquery(paper_ids: list[str], table_ref: str | None = None) -> set[str]:
    """Check which papers already exist in BigQuery."""
    if not paper_ids:
        return set()

    if not table_ref:
        table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_NAME}"

    bq_client = bigquery.Client()
    paper_id_list = ", ".join([f"'{paper_id}'" for paper_id in paper_ids])

    query = f"""
    SELECT DISTINCT paper_id
    FROM `{table_ref}`
    WHERE paper_id IN ({paper_id_list})
    """

    try:
        query_job = bq_client.query(query)
        results = list(query_job.result())
        existing_ids = {row.paper_id for row in results}
        logger.debug(f"Found {len(existing_ids)} papers already in BigQuery")
        return existing_ids
    except Exception as e:
        logger.error(f"Error checking existing papers: {e}")
        return set()


async def perform_vector_search(
    query_embedding: list[float],
    paper_ids: list[str],
    table_ref: str | None = None,
    top_k: int = 15,
    project_id: str | None = BIGQUERY_PROJECT_ID,
) -> list[dict]:
    """Perform vector search in BigQuery."""
    bq_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    if not table_ref:
        table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table_ref}"

    # Create embedding string for SQL
    embedding_str = ",".join(map(str, query_embedding))

    # Filter to papers from Elasticsearch results first, then get additional results
    paper_id_list = ", ".join([f"'{paper_id}'" for paper_id in paper_ids])

    # Get filtered results from ES papers
    filtered_sql = f"""
    SELECT
        base.text,
        base.paper_id,
        base.chunk_id,
        base.chunk_index,
        distance as similarity_score
    FROM
        VECTOR_SEARCH(
            TABLE `{table_ref}`,
            'embedding',
            (SELECT [{embedding_str}] as embedding),
            top_k => {top_k}
        )
    WHERE base.paper_id IN ({paper_id_list})
    ORDER BY similarity_score
    """

    # Get additional results from other papers
    unfiltered_sql = f"""
    SELECT
        base.text,
        base.paper_id,
        base.chunk_id,
        base.chunk_index,
        distance as similarity_score
    FROM
        VECTOR_SEARCH(
            TABLE `{table_ref}`,
            'embedding',
            (SELECT [{embedding_str}] as embedding),
            top_k => {top_k}
        )
    WHERE base.paper_id NOT IN ({paper_id_list})
    ORDER BY similarity_score
    LIMIT {top_k}
    """

    try:
        # Execute both queries
        filtered_results = await asyncio.to_thread(lambda: list(bq_client.query(filtered_sql).result()))
        unfiltered_results = await asyncio.to_thread(lambda: list(bq_client.query(unfiltered_sql).result()))

        # Combine and format results
        all_results = []
        for row in filtered_results + unfiltered_results:
            all_results.append(
                {
                    "text": row.text,
                    "paper_id": row.paper_id,
                    "chunk_id": row.chunk_id,
                    "chunk_index": row.chunk_index,
                    "similarity_score": float(row.similarity_score),
                    "source": "elasticsearch_papers" if row in filtered_results else "additional_papers",
                }
            )

        logger.debug(f"Vector search returned {len(all_results)} results")
        return all_results

    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []
