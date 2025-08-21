"""SPARQL client for querying the knowledge graph."""

import csv
import os
from dataclasses import dataclass
from io import StringIO
from typing import Any

import dotenv
import httpx
from loguru import logger

dotenv.load_dotenv()

KNOWLEDGE_GRAPH_URL = os.environ.get(
    "KNOWLEDGE_GRAPH_URL", "https://graphdb-ingress-test.centaur-platform-dev.com/repositories/cb-head"
)
KNOWLEDGE_GRAPH_TIMEOUT = int(os.environ.get("KNOWLEDGE_GRAPH_TIMEOUT", "30"))


@dataclass
class KGConfig:
    """Configuration for the knowledge graph."""

    url: str = KNOWLEDGE_GRAPH_URL
    timeout: int = KNOWLEDGE_GRAPH_TIMEOUT
    headers: dict[str, str] = {
        "Content-Type": "application/sparql-query",
        "Accept": "text/tab-separated-values",
    }
    verbose: bool = False


class GraphClient:
    """SPARQL client for querying the knowledge graph."""

    def __init__(self, config: KGConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout, headers=config.headers)

    async def _query(self, sparql: str) -> list[dict[str, str]]:
        """Execute a SPARQL SELECT query and return results."""
        try:
            response = await self.client.post(self.config.url, content=sparql.encode("utf-8"))
            response.raise_for_status()
            return list(csv.DictReader(StringIO(response.text), delimiter="\t"))
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    async def get_gene_info(self, ensembl_id: str) -> dict[str, Any] | None:
        """Get gene information by Ensembl ID."""
        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?approved_symbol ?approved_name ?synonym ?entrez_value
            WHERE {{
                ?gene a cbs:Gene ;
                    cbs:hasId "{ensembl_id}" ;
                    cbs:hasApprovedSymbol ?approved_symbol ;
                    cbs:hasApprovedName ?approved_name ;
                    cbs:identifiedBy ?entrez_id .
                OPTIONAL {{
                    ?gene cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym .
                }}
                OPTIONAL {{
                    ?entrez_id cbs:hasValue ?entrez_value .
                    FILTER(CONTAINS(STR(?entrez_id), "entrez"))
                }}
            }}"""

        results = await self._query(sparql_query)
        if not results:
            return None

        gene_info = {
            "ensembl_id": ensembl_id,
            "approved_symbol": results[0].get("?approved_symbol", ""),
            "approved_name": results[0].get("?approved_name", ""),
            "entrez_id": next((r.get("?entrez_value") for r in results if r.get("?entrez_value")), None),
            "synonyms": list({r.get("?synonym") for r in results if r.get("?synonym")}),
        }
        return gene_info

    async def get_compound_info(self, compound_id: str) -> dict[str, Any] | None:
        """Get compound information by compound ID."""
        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?name ?synonym
            WHERE {{
                ?compound a cbs:Compound ;
                    cbs:hasId "{compound_id}" .
                OPTIONAL {{ ?compound cbs:hasName ?name . }}
                OPTIONAL {{
                    ?compound cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym .
                }}
            }}"""

        results = await self._query(sparql_query)
        if not results:
            return None

        return {
            "compound_id": compound_id,
            "names": list({r.get("?name") for r in results if r.get("?name")}),
            "synonyms": list({r.get("?synonym") for r in results if r.get("?synonym")}),
        }

    async def get_disease_info(self, mondo_id: str) -> dict[str, Any] | None:
        """Get disease information by Mondo ID."""
        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?term ?type
            WHERE {{
                BIND(cbs:mondo_{mondo_id} AS ?disease)
                {{
                    ?disease cbs:hasName ?term .
                    BIND("name" as ?type)
                }}
                UNION
                {{
                    ?disease cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?term .
                    BIND("synonym" as ?type)
                }}
            }}"""

        results = await self._query(sparql_query)
        if not results:
            return None

        return {
            "mondo_id": mondo_id,
            "names": [r.get("?term") for r in results if r.get("?type") == "name" and r.get("?term")],
            "synonyms": [r.get("?term") for r in results if r.get("?type") == "synonym" and r.get("?term")],
        }

    async def get_cellular_context_info(self, context_id: str) -> dict[str, Any] | None:
        """Get cellular context information (cell types, cell lines, etc)."""
        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?name ?synonym ?context_type ?tissue ?organ
            WHERE {{
                {{
                    ?context a cbs:CellType ;
                        cbs:hasId "{context_id}" .
                    BIND("cell_type" as ?context_type)
                }}
                UNION
                {{
                    ?context a cbs:CellLine ;
                        cbs:hasId "{context_id}" .
                    BIND("cell_line" as ?context_type)
                }}
                OPTIONAL {{ ?context cbs:hasName ?name . }}
                OPTIONAL {{
                    ?context cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym .
                }}
                OPTIONAL {{ ?context cbs:hasTissue ?tissue . }}
                OPTIONAL {{ ?context cbs:hasOrgan ?organ . }}
            }}"""

        results = await self._query(sparql_query)
        if not results:
            return None

        return {
            "context_id": context_id,
            "context_type": results[0].get("?context_type"),
            "names": list({r.get("?name") for r in results if r.get("?name")}),
            "synonyms": list({r.get("?synonym") for r in results if r.get("?synonym")}),
            "tissues": list({r.get("?tissue") for r in results if r.get("?tissue")}),
            "organs": list({r.get("?organ") for r in results if r.get("?organ")}),
        }

    async def close(self):
        """Clean up HTTP client resources."""
        await self.client.aclose()
