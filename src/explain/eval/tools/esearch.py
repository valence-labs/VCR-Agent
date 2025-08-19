import asyncio
from typing import Any

from pydantic import BaseModel, Field

from explain.literature._esearch_utils import search_indexes

from ._base import ToolVerifier


class LiteratureSearchArgs(BaseModel):
    keywords: list[str] = Field(..., description="List of keywords to search for in title, abstract, and body.")
    top_k: int = Field(50, ge=1, le=200, description="Maximum number of documents to retrieve.")
    indexes: list[str] | None = Field(
        default=["full"],
        description="List of indexes to search. Available: 'full', 'fda', 'pdf', 'informa', 'informa-drugs'. Use full for full-text search",
    )


class LiteratureSearchVerifier(ToolVerifier):
    """
    Search biomedical literature across multiple Elasticsearch indexes.
    Returns a ranked list of articles (id, title, abstract, content, score) with source index information.
    """

    name = "literature_fulltext_search"
    description = (
        "Search biomedical literature by keywords across multiple indexes (full-text, FDA, PDF docs, Informa trials/drugs); "
        "returns ranked articles with abstracts, content, and source index information."
    )
    args_schema = LiteratureSearchArgs

    def _format_article(self, article: dict[str, Any]) -> str:
        """
        Format the article to include the source URL.
        """
        doc_id = article.pop("id", "")
        # Extract numeric part from _id (e.g., "pubmed_24362528" -> "24362528")
        pmid_numeric = doc_id.replace("pubmed_", "").replace("PMC", "").replace("pmc_", "")
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_numeric}/" if pmid_numeric else "N/A"
        article["source"] = source_url
        return article

    def _tool_logic(self, args: LiteratureSearchArgs) -> tuple[float, dict[str, Any]]:
        try:
            # Perform multi-index search
            articles = asyncio.run(search_indexes(keywords=args.keywords, indexes=args.indexes, top_k=args.top_k))

            real_hits = [a for a in articles if a.get("id") != "fallback_id"]
            n = len(real_hits)
            reward = 1.0 if n > 0 else 0.0
            # Count articles by source index for reporting
            index_counts = {}
            for article in real_hits:
                source_index = article.get("source_index", "unknown")
                index_counts[source_index] = index_counts.get(source_index, 0) + 1

            return reward, {
                "keywords": args.keywords,
                "indexes_searched": args.indexes,
                "total_count": len(articles),
                "results": real_hits,
            }
        except Exception as e:
            return 0.0, {"error": f"Literature search failed: {e}"}
