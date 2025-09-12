import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from explain.literature.evidence_query import process_literature_query

from ._base import ToolVerifier


class EvidenceSearchArgs(BaseModel):
    query: str = Field(..., description="Scientific question to answer using literature evidence")
    keywords: list[str] = Field(..., description="Keywords to search for relevant papers and evidence")
    top_k: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum papers to retrieve for finding evidence. Large values can lead to slower response times, but more relevant results.",
    )
    vector_top_k: int = Field(
        15,
        ge=1,
        le=50,
        description="Maximum number of snippet passages with similarity to the query to use for answer generation.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for semantic similarity search of evidence",
    )
    llm_model: str = Field(
        default="gpt-5",
        description="Language model for generating evidence-based answers",
    )
    evidence_only: bool = Field(
        default=True,
        description="If True, answer strictly from evidence only. If False, can supplement with scientific knowledge.",
    )


class Evidencer(ToolVerifier):
    """
    Scientific question answering tool powered by literature evidence.

    This tool answers scientific questions by finding supporting evidence from biomedical literature,
    then generating clear, well-founded answers based on that evidence. The process includes:

    1. Search relevant papers using keywords (Elasticsearch)
    2. Find semantically similar content using vector search (BigQuery)
    3. Generate evidence-supported answers from the most relevant findings

    Answers are grounded in literature and can be configured to use only the retrieved evidence
    or to supplement with the model's scientific knowledge when helpful.
    """

    name = "literature_evidence_answerer"
    description = (
        "Generate evidence-supported answers to scientific questions using literature search. "
        "Searches pubmed using elasticsearch and bigquery vector search to find relevant evidence, "
        "then synthesizes findings into clear, scientifically-grounded answers. "
        "Answers can be strictly evidence-only or enhanced with scientific knowledge."
    )
    args_schema = EvidenceSearchArgs

    def _tool_logic(self, args: EvidenceSearchArgs) -> tuple[float, dict[str, Any]]:
        try:
            return asyncio.run(self._async_tool_logic(args))
        except Exception as e:
            logger.error(f"Error in evidence-based answer generation: {e}")
            return 0.0, {"error": f"Evidence-based answering failed: {e}"}

    async def _async_tool_logic(self, args: EvidenceSearchArgs) -> tuple[float, dict[str, Any]]:
        """Generate evidence-supported answers to scientific questions."""
        try:
            response = await process_literature_query(
                query=args.query,
                keywords=args.keywords,
                top_k=args.top_k,
                vector_top_k=args.vector_top_k,
                indexes=["full"],
                embedding_model=args.embedding_model,
                llm_model=args.llm_model,
                generate_answer=True,
                evidence_only=args.evidence_only,
            )

            response.pop("results", None)
            # Calculate reward based on whether we found evidence and generated an answer
            has_evidence = response.get("num_elasticsearch_results", 0) > 0
            has_answer = response.get("generated_answer", None) is not None
            reward = 1.0 if (has_evidence and has_answer) else 0.5 if has_answer else 0.0

            return reward, response
        except Exception as e:
            logger.error(f"Error in evidence-based answer generation: {e}")
            return 0.0, {"error": f"Evidence-based answering failed: {e}"}
