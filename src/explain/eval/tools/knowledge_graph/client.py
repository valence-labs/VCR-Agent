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


# Configuration
KNOWLEDGE_GRAPH_URL = os.environ.get(
    "KNOWLEDGE_GRAPH_URL", "https://graphdb-ingress-test.centaur-platform-dev.com/repositories/cb-head"
)  # noqa: S105
KNOWLEDGE_GRAPH_TIMEOUT = int(os.environ.get("KNOWLEDGE_GRAPH_TIMEOUT", "30"))  # noqa: S105


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

        if self.config.verbose:
            logger.debug(f"Executing SPARQL query: {sparql}")

        try:
            response = await self.client.post(self.config.url, content=sparql.encode("utf-8"))
            response.raise_for_status()
            body = response.text
            return list(csv.DictReader(StringIO(body), delimiter="\t"))

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in SPARQL query: {e}")
            raise Exception(f"Knowledge graph query failed: {e}") from e
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise Exception(f"Failed to execute knowledge graph query: {e}") from e

    async def get_gene_info(self, ensembl_id: str) -> list[dict[str, str]] | None:
        """
        Query the knowledge graph for gene information by Ensembl ID.

        Args:
            ensembl_id: Ensembl gene ID (e.g., "ENSG00000135503")

        Returns:
            Dictionary with gene information or None if not found:
            {
                'ensembl_id': str,
                'entrez_id': str or None,
                'approved_symbol': str,
                'approved_name': str,
                'synonyms': List[str]
            }
        """
        if self.config.verbose:
            logger.debug(f"Querying gene info for: {ensembl_id}")

        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?approved_symbol ?approved_name ?synonym ?entrez_value
            WHERE {{
                ?gene a cbs:Gene ;
                    cbs:hasId ?ensembl_id ;
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
                FILTER(?ensembl_id = "{ensembl_id}")
            }}
            """

        try:
            results = await self._query(sparql_query)

            if not results:
                if self.config.verbose:
                    logger.warning(f"No gene information found for: {ensembl_id}")
                return None

            # Process results - group synonyms and get other info
            gene_info: dict[str, str] = {
                "ensembl_id": ensembl_id,
                "entrez_id": None,
                "approved_symbol": "",
                "approved_name": "",
                "synonyms": [],
            }

            # Extract information from results
            for result in results:
                # Get basic gene info from first result
                if not gene_info["approved_symbol"]:
                    gene_info["approved_symbol"] = result.get("?approved_symbol", "")
                    gene_info["approved_name"] = result.get("?approved_name", "")

                # Get entrez ID if available
                if result.get("?entrez_value") and not gene_info["entrez_id"]:
                    gene_info["entrez_id"] = result.get("?entrez_value")

                # Collect synonyms
                synonym = result.get("?synonym")
                if synonym and synonym not in gene_info["synonyms"]:
                    gene_info["synonyms"].append(synonym)

            if self.config.verbose:
                logger.debug(f"Retrieved gene info for {ensembl_id}: {gene_info['approved_symbol']}")

            return gene_info

        except Exception as e:
            logger.error(f"Error querying gene info for {ensembl_id}: {e}")
            return None

    async def get_compound_info(self, compound_id: str) -> list[dict[str, str]] | None:
        """
        Query the knowledge graph for compound information by compound ID.
        """

        sparql_query = """
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?name ?synonym
            WHERE {
                ?compound a cbs:Compound ;

                ?compound cbs:hasName ?name ;
                    cbs:hasSynonym ?synonym .
            }
            """

        results = await self._query(sparql_query)
        if self.config.verbose:
            logger.debug(f"Query for '{compound_id}' returned {len(results)} results")

        if not results:
            if self.config.verbose:
                logger.warning(f"No compound information found for: {compound_id}")
            return None

        compound_info: dict[str, Any] = {
            "compound_id": compound_id,
            "names": [],
            "synonyms": [],
        }

        for result in results:
            name = result.get("?name")
            synonym = result.get("?synonym")

            if name:
                compound_info["names"].append(name)
            if synonym and synonym not in compound_info["synonyms"]:
                compound_info["synonyms"].append(synonym)

        if self.config.verbose:
            logger.debug(f"Retrieved compound info for {compound_id}: {compound_info['names'][0]}")

    async def get_disease_info_by_mondo_id(self, mondo_id: str) -> list[dict[str, str]] | None:
        """
        Query the knowledge graph for disease information by Mondo ID.

        Args:
            mondo_id: Mondo disease ID (e.g., "0007254")

        Returns:
            Dictionary with disease information or None if not found:
            {
                'mondo_id': str,
                'names': List[str],
                'synonyms': List[str]
            }
        """

        if self.config.verbose:
            logger.debug(f"Querying disease info for: {mondo_id}")

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
            }}
            ORDER BY ?type ?term
            """

        try:
            results = await self._query(sparql_query)
            if self.config.verbose:
                logger.debug(f"Query for '{mondo_id}' returned {len(results)} results")

            if not results:
                if self.config.verbose:
                    logger.warning(f"No disease information found for: {mondo_id}")
                return None

            # Process results - separate names and synonyms
            disease_info: dict[str, Any] = {
                "mondo_id": mondo_id,
                "names": [],
                "synonyms": [],
            }

            # Extract information from results
            for result in results:
                term = result.get("?term")
                term_type = result.get("?type")

                if term:
                    if term_type == "name" and term not in disease_info["names"]:
                        disease_info["names"].append(term)
                    elif term_type == "synonym" and term not in disease_info["synonyms"]:
                        disease_info["synonyms"].append(term)

            # Log warnings for missing context
            if not disease_info["names"]:
                if self.config.verbose:
                    logger.warning(f"No disease names found for {mondo_id} - this may affect analysis quality")

            if not disease_info["synonyms"]:
                if self.config.verbose:
                    logger.warning(f"No disease synonyms found for {mondo_id} - this may limit search scope")

            primary_name = disease_info["names"][0] if disease_info["names"] else "Unknown"
            if self.config.verbose:
                logger.debug(
                    f"Retrieved disease info for {mondo_id}: {primary_name} "
                    f"({len(disease_info['names'])} names, "
                    f"{len(disease_info['synonyms'])} synonyms)"
                )

            return disease_info

        except Exception as e:
            if self.config.verbose:
                logger.error(f"Error querying disease info for {mondo_id}: {e}")
            return None

    async def close(self):
        """Clean up HTTP client resources."""
        if self.config.verbose:
            logger.debug("Closing Knowledge Graph client")
        await self.client.aclose()
