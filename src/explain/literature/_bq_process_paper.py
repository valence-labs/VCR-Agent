import asyncio
import io
import os
import uuid
from math import ceil

import dotenv
import fastavro
import tiktoken
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, storage
from litellm import aembedding as litellm_aembedding
from loguru import logger

dotenv.load_dotenv()

BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "rxrx-medchem-auto-dev")
BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "medchem_auto_dev_bigquery_db")
BIGQUERY_TABLE_NAME = os.getenv("BIGQUERY_TABLE_NAME", "hooke_elastic_search_index")
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME", "hooke-literature-test")
VECTOR_INDEX_NAME = f"{BIGQUERY_TABLE_NAME}_embedding_idx"
TABLE_SCHEMA = [
    bigquery.SchemaField("text", field_type="STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_id", field_type="STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_index", field_type="INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("total_chunks", field_type="INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("paper_id", field_type="STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", field_type="FLOAT64", mode="REPEATED"),
]


def chunk_text(text: str, chunk_size: int, overlap_size: int, dockey: str) -> list[dict]:
    try:
        content = text
        enc = tiktoken.get_encoding("cl100k_base")
        content_tokens = enc.encode(content)
        if not content_tokens:
            return []
        char_count = len(content)
        token_count = len(content_tokens)
        chars_per_token = char_count / token_count if token_count > 0 else 1
        chunk_tokens = chunk_size / chars_per_token if chars_per_token > 0 else 0
        overlap_tokens = overlap_size / chars_per_token if chars_per_token > 0 else 0
        chunk_count = ceil(token_count / chunk_tokens) if chunk_tokens > 0 else 1
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


def chunk_papers(paper_content: str, paper_id: str, chunk_size: int = 4096, overlap_size: int = 128) -> list:
    chunks = chunk_text(
        text=paper_content,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        dockey=paper_id,
    )
    for chunk in chunks:
        chunk["paper_id"] = paper_id
    return chunks


async def aembed_chunks(chunks: list, embedding_model: str = "text-embedding-3-small", batch_size: int = 100) -> list:
    if not chunks:
        return []

    try:
        # Optimize batch size based on content length
        avg_length = sum(len(chunk.get("text", "")) for chunk in chunks) / len(chunks)
        if avg_length > 2000:  # For longer texts, use smaller batches
            batch_size = min(batch_size, 50)

        tasks = []
        batch_info = []  # Track batch start indices for result mapping

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            # Create a coroutine for the API call but don't await it yet
            task = litellm_aembedding(model=embedding_model, input=texts)
            tasks.append(task)
            batch_info.append(i)  # Store start index

        # Run all tasks concurrently
        logger.info(f"Running {len(tasks)} embedding batches concurrently (avg_length: {avg_length:.0f} chars)...")
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process the results more efficiently
        embedded_chunks = []
        for batch_idx, (response, start_idx) in enumerate(zip(responses, batch_info, strict=True)):
            if isinstance(response, Exception):
                logger.error(f"Batch {batch_idx} failed: {response}")
                # Add chunks without embeddings for this batch
                batch_end = min(start_idx + batch_size, len(chunks))
                embedded_chunks.extend(chunks[start_idx:batch_end])
                continue

            embeddings = [item["embedding"] for item in response.data]
            batch_end = min(start_idx + batch_size, len(chunks))

            for chunk_idx, embedding in zip(range(start_idx, batch_end), embeddings, strict=True):
                embedded_chunk = chunks[chunk_idx].copy()
                embedded_chunk["embedding"] = embedding
                embedded_chunks.append(embedded_chunk)

        logger.info(
            f"Successfully embedded {len([c for c in embedded_chunks if 'embedding' in c])}/{len(chunks)} chunks"
        )
        return embedded_chunks
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        return chunks


async def aload_to_bigquery(
    embedded_chunks: list,
    table_name: str = BIGQUERY_TABLE_NAME,
    project_id: str = BIGQUERY_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
    build_index: bool = False,
    use_gcs: bool = False,
):
    """Uploads embedded chunks to BigQuery using streaming inserts."""
    table_ref_string = f"{project_id}.{dataset_id}.{table_name}"
    await ensure_dataset_exists(project_id, dataset_id)
    await ensure_table_exists(table_name, TABLE_SCHEMA, project_id, dataset_id, rebuild=False)
    bq_client = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(VECTOR_BUCKET_NAME)

    def _df_to_avro_bytes(chunks):
        """Convert a Pandas DataFrame to an Avro file in memory."""
        avro_schema = {
            "type": "record",
            "name": "embedding_chunk",
            "fields": [
                {"name": "text", "type": "string"},
                {"name": "chunk_id", "type": "string"},
                {"name": "chunk_index", "type": "long"},
                {"name": "total_chunks", "type": "long"},
                {"name": "paper_id", "type": "string"},
                {"name": "embedding", "type": {"type": "array", "items": "double"}},
            ],
        }

        buf = io.BytesIO()
        fastavro.writer(buf, avro_schema, chunks)
        buf.seek(0)
        return buf

    try:
        unique_filename = f"bq_load/{uuid.uuid4()}.parquet"
        buf = await asyncio.to_thread(_df_to_avro_bytes, embedded_chunks)
        blob = bucket.blob(unique_filename)
        blob.upload_from_file(buf, rewind=True, content_type="application/octet-stream")
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.AVRO,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        logger.info(f"Starting BigQuery batch load from {unique_filename}...")
        load_job = await asyncio.to_thread(
            bq_client.load_table_from_uri,
            f"gs://{VECTOR_BUCKET_NAME}/{unique_filename}",
            table_ref_string,
            job_config=job_config,
        )
        await asyncio.to_thread(load_job.result)  # Wait for the job to complete

        logger.debug(f"Successfully loaded {load_job.output_rows} rows via GCS to {table_ref_string}.")
        blob = bucket.blob(unique_filename)
        blob.delete()
        logger.debug(f"Deleted temporary Parquet file from {unique_filename}.")

        if build_index:
            try:
                # EN: this can fail since we need at least 5k rows to build the index.
                await ensure_vector_index_exists(table_ref_string=table_ref_string, project_id=project_id)
            except Exception as e:
                logger.error(f"Error building vector index: {e}")
            logger.debug("Vector index build process finished.")

    except Exception as e:
        logger.error(f"Error in BigQuery load process: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


async def check_existing_papers_in_bigquery(paper_ids: list[str], table_ref: str | None = None) -> set[str]:
    """Check which papers already exist in BigQuery."""
    if not paper_ids:
        return set()
    if not table_ref:
        table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_NAME}"

    bq_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    paper_id_list = ", ".join([f"'{paper_id}'" for paper_id in paper_ids])
    query = f"SELECT DISTINCT paper_id FROM `{table_ref}` WHERE paper_id IN ({paper_id_list})"

    try:
        query_job = await asyncio.to_thread(bq_client.query, query)
        results = await asyncio.to_thread(list, query_job.result())
        existing_ids = {row.paper_id for row in results}
        logger.debug(f"Found {len(existing_ids)} papers already in BigQuery")
        return existing_ids
    except NotFound:
        logger.debug(f"Table {table_ref} does not exist yet")
        return set()
    except Exception as e:
        logger.error(f"Error checking existing papers: {e}")
        return set()


async def perform_vector_search(
    query_embedding: list[float],
    paper_ids: list[str],
    table_ref: str | None = None,
    top_k: int = 15,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """Perform vector search in BigQuery using a single, efficient query."""
    bq_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    if not table_ref:
        table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_NAME}"

    # More efficient embedding handling - avoid string conversion
    embedding_list_str = "[" + ",".join(map(str, query_embedding)) + "]"
    paper_id_list = ", ".join([f"'{pid}'" for pid in paper_ids]) if paper_ids else "''"

    threshold_clause = ""
    if similarity_threshold is not None:
        max_distance = 1.0 - similarity_threshold
        threshold_clause = f"WHERE distance <= {max_distance}"

    sql_query = f"""
    SELECT
        base.text, base.paper_id, base.chunk_id, base.chunk_index,
        1 - distance AS similarity_score,
        IF(base.paper_id IN ({paper_id_list}), 'elasticsearch_papers', 'additional_papers') AS source
    FROM
        VECTOR_SEARCH(
            TABLE `{table_ref}`, 'embedding', (SELECT {embedding_list_str} as embedding),
            top_k => {top_k}, distance_type => 'COSINE'
        )
    {threshold_clause}
    ORDER BY distance ASC
    LIMIT {top_k}
    """
    try:
        query_job = await asyncio.to_thread(bq_client.query, sql_query)
        results_df = await asyncio.to_thread(query_job.to_dataframe)

        all_results = results_df.to_dict("records")
        logger.debug(f"Vector search returned {len(all_results)} results")
        return all_results
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []


async def ensure_dataset_exists(
    project_id: str = BIGQUERY_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
    location: str = "US",
) -> None:
    """Ensure that a BigQuery dataset exists, creating it if necessary."""
    bq_client = bigquery.Client(project=project_id)
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        await asyncio.to_thread(bq_client.get_dataset, dataset_ref)
        logger.debug(f"Dataset {dataset_ref} already exists")
    except NotFound:
        try:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            await asyncio.to_thread(bq_client.create_dataset, dataset)
            logger.info(f"Created dataset {dataset_ref}")
        except Exception as e:
            logger.error(f"Failed to create dataset {dataset_ref}: {e}")
            raise


async def ensure_table_exists(
    table_name: str = BIGQUERY_TABLE_NAME,
    schema: list[bigquery.SchemaField] = TABLE_SCHEMA,
    project_id: str = BIGQUERY_PROJECT_ID,
    dataset_id: str = BIGQUERY_DATASET_ID,
    rebuild: bool = False,
) -> bigquery.Table:
    """Ensure that a BigQuery table exists, using sync client with asyncio."""
    bq_client = bigquery.Client(project=project_id)
    table_ref_string = f"{project_id}.{dataset_id}.{table_name}"

    if rebuild:
        try:
            await asyncio.to_thread(bq_client.delete_table, table_ref_string)
            logger.info(f"Deleted existing table {table_ref_string} for rebuild")
        except NotFound:
            logger.debug(f"Table {table_ref_string} not found during rebuild (expected).")

    try:
        table = await asyncio.to_thread(bq_client.get_table, table_ref_string)
        logger.debug(f"Table {table_ref_string} already exists")
        return table
    except NotFound:
        try:
            table = bigquery.Table(table_ref_string, schema=schema)
            created_table = await asyncio.to_thread(bq_client.create_table, table)
            action = "Rebuilt" if rebuild else "Created"
            logger.info(f"{action} table {created_table.table_id}")
            return created_table
        except Exception as e:
            logger.error(f"Failed to create table {table_ref_string}: {e}")
            raise


async def ensure_vector_index_exists(
    table_ref_string: str = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_NAME}",
    index_name: str = VECTOR_INDEX_NAME,
    project_id: str = BIGQUERY_PROJECT_ID,
) -> None:
    """Checks for and creates a vector index if it doesn't exist."""
    bq_client = bigquery.Client(project=project_id)
    check_index_query = f"""
    SELECT index_name
    FROM `{table_ref_string.rsplit(".", 1)[0]}.INFORMATION_SCHEMA.VECTOR_INDEXES`
    WHERE table_name = '{table_ref_string.split(".")[-1]}' AND index_name = '{index_name}'
    """
    try:
        query_job = await asyncio.to_thread(bq_client.query, check_index_query)
        results = await asyncio.to_thread(list, query_job.result())

        if results:
            logger.debug(f"Vector index '{index_name}' already exists.")
            return

        logger.info(f"Vector index '{index_name}' not found. Creating it now...")
        create_index_ddl = f"""
        CREATE VECTOR INDEX {index_name}
        ON `{table_ref_string}`(embedding)
        OPTIONS(index_type = 'IVF', distance_type = 'COSINE', ivf_options = '{{"num_lists": 500}}')
        """
        create_job = await asyncio.to_thread(bq_client.query, create_index_ddl)
        await asyncio.to_thread(create_job.result)
        logger.info(f"Successfully created vector index '{index_name}'.")

    except Exception as e:
        if "already exists" in str(e):
            logger.debug(f"Vector index '{index_name}' was created by another process.")
        else:
            raise
