"""SPARQL client for querying the knowledge graph."""

import csv
import os
from io import StringIO

import aiohttp

# Configuration
DEFAULT_KNOWLEDGE_GRAPH_URL = "https://graphdb-ingress-test.centaur-platform-dev.com/repositories/cb-head"
KNOWLEDGE_GRAPH_URL = os.environ.get("KNOWLEDGE_GRAPH_URL", DEFAULT_KNOWLEDGE_GRAPH_URL)
KNOWLEDGE_GRAPH_TIMEOUT = int(os.environ.get("KNOWLEDGE_GRAPH_TIMEOUT", "30"))


class GraphClient:
    """SPARQL client for querying the knowledge graph."""

    def __init__(self, url: str = KNOWLEDGE_GRAPH_URL, timeout: int = KNOWLEDGE_GRAPH_TIMEOUT):
        self.url = url
        self.timeout = timeout

    async def query(self, sparql: str) -> list[dict[str, str]]:
        """Execute a SPARQL SELECT query and return results."""
        headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "text/tab-separated-values",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.url, data=sparql, headers=headers) as response:
                    body = await response.text()

                    if response.status >= 400:
                        raise RuntimeError(f"SPARQL query failed with status {response.status}: {body}")

                    # Parse TSV response
                    return list(csv.DictReader(StringIO(body), delimiter="\t"))

        except TimeoutError as exc:
            raise RuntimeError(f"SPARQL query timed out after {self.timeout} seconds") from exc
        except (aiohttp.ClientError, OSError) as e:
            raise RuntimeError(f"SPARQL query failed: {str(e)}") from e

    async def ask(self, sparql: str) -> bool:
        """Execute a SPARQL ASK query and return boolean result."""
        headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.url, data=sparql, headers=headers) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(f"SPARQL ASK query failed with status {response.status}: {body}")

                    result = await response.json()
                    return result.get("boolean", False)

        except TimeoutError as exc:
            raise RuntimeError(f"SPARQL ASK query timed out after {self.timeout} seconds") from exc
        except (aiohttp.ClientError, OSError) as e:
            raise RuntimeError(f"SPARQL ASK query failed: {str(e)}") from e
