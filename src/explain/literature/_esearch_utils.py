## A good chunk of the following code has been adapted from
# [doctor-retriever](https://github.com/recursionpharma/doctor-retriever/tree/trunk)

import asyncio
import os
from itertools import chain

import dotenv
from aiolimiter import AsyncLimiter
from elasticsearch import AsyncElasticsearch
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._const import BASE_DELAY, KEYWORDS, MAP_TO_KEYWORDS, MAX_DELAY, MAX_RETRIES

dotenv.load_dotenv()


class NoopLimiter:
    """A no-op rate limiter for when rate limiting is not needed."""

    def __init__(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def locked(self):
        return False

    async def acquire(self):
        return True

    def release(self):
        pass

    def get_value(self):
        return float("inf")


class ElasticSearchIndex:
    """Basic search client for Elasticsearch"""

    INDEXES = {
        "full": "full-text-articles",
        "fda": "fda-prescribing-labels-v2",
        "pdf": "pdf-docs-v3",
        "informa": "informa-trials-v4",
        "informa-drugs": "informa-drugs-v1",
    }

    URL = os.getenv("FULLTEXT_ELASTICSEARCH_URL", "https://elasticsearch.centaur-platform-dev.com:443")

    def __init__(
        self,
        url=URL,
        index="full",
        rate_limit: int | None = None,
    ):
        index = self.INDEXES.get(index, index)
        self.__client = AsyncElasticsearch(
            hosts=[url],
            max_retries=20,
            retry_on_timeout=True,
            request_timeout=120,
        )
        self.__index = index
        self.__rate_limiter = AsyncLimiter(rate_limit) if isinstance(rate_limit, int) else NoopLimiter()

    async def close(self):
        await self.__client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *execinfo):
        await self.__client.close()

    async def search(self, body):
        async with self.__rate_limiter:
            return await self.__client.search(index=self.__index, **body)

    async def create_search_body(
        self,
        keywords: list[str],
        primitive_catalog: str | None = "full",
        top_k: int = 50,
        gene_id: str | None = None,
        disease_id: str | None = None,
        fragment_size: int | None = 1000,
        number_of_fragments: int | None = 4,
    ) -> dict:
        """
        Constructs an Elasticsearch search body using nested queries for contextual precision.
        Args:
            keywords: List of keywords to search for
            primitive_catalog: Type of input to search for
            top_k: Maximum number of results
            gene_id: Gene ID to search for
            disease_id: Disease ID to search for
            fragment_size: Size of the fragment to highlight
            number_of_fragments: Number of fragments to highlight
        """

        # 1. Merge relevant keyword groups
        group_names = MAP_TO_KEYWORDS.get(primitive_catalog, [])
        keyword_list = list(chain.from_iterable(KEYWORDS[group]["keywords"] for group in group_names))

        # 2. Build `should` clauses from all search terms
        should_clauses = [{"match_phrase": {"sections.text": {"query": term, "boost": 5}}} for term in keywords]
        should_clauses.extend(
            [{"match_phrase": {"sections.text": {"query": term, "boost": 1}}} for term in set(keyword_list)]
        )
        nested_query = {
            "nested": {
                "path": "sections",
                "query": {"bool": {"must": []}},
                "inner_hits": {
                    "highlight": {
                        "fields": {
                            "sections.text": {
                                "fragment_size": fragment_size or 1000,
                                "number_of_fragments": number_of_fragments,
                                "require_field_match": True,
                            }
                        },
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"],
                    }
                },
            }
        }

        # Add gene and disease filters if provided
        if gene_id:
            nested_query["nested"]["query"]["bool"]["must"].append({"term": {"sections.spans.id": {"value": gene_id}}})
        if disease_id:
            nested_query["nested"]["query"]["bool"]["must"].append(
                {"term": {"sections.spans.parent_ids": {"value": disease_id}}}
            )

        nested_query["nested"]["query"]["bool"]["must"].append(
            {"bool": {"should": should_clauses, "minimum_should_match": 1}}
        )

        search_body = {
            "size": top_k,
            "_source": True,
            "query": {"bool": {"must": [nested_query], "must_not": [{"exists": {"field": "retraction_reasons"}}]}},
        }

        return search_body


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=BASE_DELAY, max=MAX_DELAY),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _search_with_retry(client, search_body):
    """Perform elasticsearch search with retry logic."""
    return await client.search(body=search_body)


async def _search_single_index(
    keywords: list[str],
    index_name: str,
    *,
    top_k: int = 50,
    primitive_catalog: str | None = "full",
    gene_id: str | None = None,
    disease_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """
    Search a single Elasticsearch index and return articles with source index information.

    Args:
        index_name: Name of the index to search
        keywords: List of keywords to search for
        top_k: Maximum number of results
        primitive_catalog: Category of primitive catalog to augment keyword search with
        gene_id: Gene ID to search for
        disease_id: Disease ID to search for
        **kwargs: Additional keyword arguments to pass to the search body

    Returns:
        List of articles from the index with source_index field added
    """
    try:
        async with ElasticSearchIndex(index=index_name) as client:
            search_body = await client.create_search_body(
                keywords=keywords,
                top_k=top_k,
                primitive_catalog=primitive_catalog,
                gene_id=gene_id,
                disease_id=disease_id,
                **kwargs,
            )
            articles = await retrieve_article(client, search_body)

            # Add source index to each article
            for article in articles:
                if article.get("id") != "fallback_id":
                    article["source_index"] = index_name

            return articles

    except Exception as e:
        logger.warning(f"Error searching index '{index_name}': {e}")
        return []


async def retrieve_article(client: ElasticSearchIndex, search_body: dict) -> list[dict]:
    """
    Retrieve articles from ElasticSearch using an already initialized client.

    Args:
        client: Initialized ElasticSearchIndex client
        search_body: Elasticsearch query body

    Returns:
        List of articles with id, title, abstract, content, and score
    """
    retrieved_articles = []

    try:
        response = await _search_with_retry(client, search_body)

        hits = response.get("hits", {}).get("hits", [])
        logger.debug(f"Retrieved {len(hits)} hits")

        for _, hit in enumerate(hits):
            paper_id = hit.get("_id", "unknown_id")
            source = hit.get("_source", {})
            title = source.get("title", "No title provided")

            abstract = ""
            for section in source.get("sections", []):
                if section.get("type") == "ABSTRACT":
                    abstract = section.get("text", "")
                    break

            main_content = []
            for section in source.get("sections", []):
                section_type = section.get("type", "").upper()
                section_title = section.get("heading", "").upper()

                if any(
                    term in section_type or term in section_title
                    for term in [
                        "ABSTRACT",
                        "ACKNOWLEDGMENT",
                        "REFERENCE",
                        "FUNDING",
                        "AUTHOR",
                        "CONFLICT",
                        "DISCLOSURE",
                        "APPENDIX",
                    ]
                ):
                    continue

                section_text = section.get("text", "")
                if section_text:
                    main_content.append(section_text)

            # EN keeping this as is for compat, but ". " would be better
            content = " ".join(main_content)

            retrieved_articles.append(
                {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "content": content,
                    "score": hit.get("_score", 0),
                }
            )

        paper_ids = [article["id"] for article in retrieved_articles]
        logger.debug(f"Retrieved paper IDs: {paper_ids}")

    except Exception as e:
        logger.error(f"Error performing search: {e}")
        logger.exception("Full exception details:")

    if not retrieved_articles:
        logger.warning("No articles found. Adding fallback article.")
        retrieved_articles.append(
            {
                "id": "fallback_id",
                "title": "No relevant articles found",
                "abstract": "The search did not return any relevant scientific articles for the given query.",
                "content": "",
                "score": 0,
            }
        )

    return retrieved_articles


async def search_indexes(
    keywords: list[str],
    indexes: list[str] | None = None,
    top_k: int = 50,
    primitive_catalog: str | None = "full",
    gene_id: str | None = None,
    disease_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """
    Search multiple Elasticsearch indexes in parallel and concatenate results.

    Args:
        keywords: List of keywords to search for
        indexes: List of index names to search in
        top_k: Maximum number of results per index
        primitive_catalog: Type of input to search for
        gene_id: Gene ID to search for
        disease_id: Disease ID to search for
        **kwargs: Additional keyword arguments to pass to the search body

    Returns:
        List of articles from all indexes combined, deduplicated by ID and sorted by score
    """
    if indexes is None:
        indexes = ["full"]

    # Run searches in parallel
    search_tasks = [
        _search_single_index(
            keywords=keywords,
            index_name=index_name,
            top_k=top_k,
            primitive_catalog=primitive_catalog,
            gene_id=gene_id,
            disease_id=disease_id,
            **kwargs,
        )
        for index_name in indexes
    ]
    index_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Combine and deduplicate results
    all_articles = []
    seen_ids = set()

    for i, articles in enumerate(index_results):
        if isinstance(articles, Exception):
            logger.warning(f"Error searching index '{indexes[i]}': {articles}")
            continue

        for article in articles:
            article_id = article.get("id")
            if article_id and article_id not in seen_ids and article_id != "fallback_id":
                all_articles.append(article)
                seen_ids.add(article_id)

    # Sort by score (descending)
    all_articles.sort(key=lambda x: x.get("score", 0), reverse=True)

    # If no real results found, add fallback
    if not all_articles:
        all_articles.append(
            {
                "id": "fallback_id",
                "title": "No relevant articles found",
                "abstract": "The search did not return any relevant scientific articles for the given query.",
                "content": "",
                "score": 0,
                "source_index": "fallback",
            }
        )

    return all_articles
