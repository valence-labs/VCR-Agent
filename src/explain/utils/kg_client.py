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
    "KNOWLEDGE_GRAPH_URL", "https://graphdb-ingress-test.centaur-platform.com/repositories/cb-head"
)
KNOWLEDGE_GRAPH_TIMEOUT = int(os.environ.get("KNOWLEDGE_GRAPH_TIMEOUT", "30"))


@dataclass
class KGConfig:
    """Configuration for the knowledge graph."""

    url: str = KNOWLEDGE_GRAPH_URL
    timeout: int = KNOWLEDGE_GRAPH_TIMEOUT
    verbose: bool = False


class KGClient:
    """SPARQL client for querying the knowledge graph."""

    def __init__(self, config: KGConfig | None = None):
        if config is None:
            config = KGConfig()
        self.config = config
        self._headers: dict[str, str] = {
            "Content-Type": "application/sparql-query",
            "Accept": "text/tab-separated-values",
        }
        self.client = httpx.AsyncClient(timeout=config.timeout, headers=self._headers)

    async def query(self, sparql: str) -> list[dict[str, str]]:
        """Execute a SPARQL SELECT query and return results."""
        try:
            response = await self.client.post(self.config.url, content=sparql.encode("utf-8"))
            response.raise_for_status()
            return list(csv.DictReader(StringIO(response.text), delimiter="\t"))
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    async def close(self):
        """Clean up HTTP client resources."""
        await self.client.aclose()

    async def get_gene_info(self, ensembl_id: str) -> dict[str, Any] | None:
        """Get gene information by Ensembl ID."""
        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?symbol ?name ?synonym ?hgnc ?entrez ?refseq ?uniprot_id
            WHERE {{
                ?gene a cbs:Gene ;
                    cbs:hasId "{ensembl_id}" ;
                    cbs:hasApprovedSymbol ?symbol ;
                    cbs:hasApprovedName ?name .
                
                OPTIONAL {{
                    ?gene cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym .
                }}

                OPTIONAL {{
                    ?gene cbs:identifiedBy ?hgnc_assertion .
                    ?hgnc_assertion cbs:hasValue ?hgnc .
                    FILTER(CONTAINS(str(?hgnc_assertion), "hgnc"))
                }}

                OPTIONAL {{
                    ?gene cbs:identifiedBy ?entrez_assertion .
                    ?entrez_assertion cbs:hasValue ?entrez .
                    FILTER(CONTAINS(str(?entrez_assertion), "entrez"))
                }}

                OPTIONAL {{
                    ?gene cbs:identifiedBy ?refseq_assertion .
                    ?refseq_assertion cbs:hasValue ?refseq .
                    FILTER(CONTAINS(str(?refseq_assertion), "refseq"))
                }}

                OPTIONAL {{
                    ?gene cbs:encodesProtein ?protein .
                    ?protein cbs:identifiedBy ?uniprot_assertion .
                    ?uniprot_assertion cbs:hasValue ?uniprot_id .
                }}
            }}"""

        results = await self.query(sparql_query)
        if not results:
            return None

        gene_info = {
            "id": ensembl_id,
            "symbol": results[0].get("?symbol", ""),
            "name": results[0].get("?name", ""),
            "entrez_id": next((r.get("?entrez") for r in results if r.get("?entrez")), None),
            "refseq_id": next((r.get("?refseq") for r in results if r.get("?refseq")), None),
            "uniprot_id": next((r.get("?uniprot_id") for r in results if r.get("?uniprot_id")), None),
            "hgnc_id": next((r.get("?hgnc") for r in results if r.get("?hgnc")), None),
            "synonyms": list({r.get("?synonym") for r in results if r.get("?synonym")}),
        }
        return gene_info

    async def get_compound_info(self, compound_id: str) -> dict[str, Any] | None:
        """Get compound information by compound ID.
        Args:
            compound_id has to be a Chembl ID (e.g. CHEMBL1234567)

        Returns:
            dict: Compound information
        """
        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT
                   ?smiles
                   ?primary_name
                   ?synonym
                   (GROUP_CONCAT(DISTINCT CONCAT(?identifier_source_name, ":", ?identifier_value); separator="|") as ?all_identifiers)
            WHERE {{
                BIND(IRI(CONCAT(str(cbs:), "compound_", "{compound_id}")) AS ?compound_uri)
                
                ?compound_uri a cbs:Compound .

                # Get SMILES - single value
                OPTIONAL {{ ?compound_uri cbs:hasCanonicalSmiles ?smiles }}

                # Get primary name and synonyms
                OPTIONAL {{ ?compound_uri cbs:hasPrimaryName ?primary_name }}
                OPTIONAL {{
                    ?compound_uri cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym .
                }}

                # Get ALL identifiers
                OPTIONAL {{
                    ?compound_uri cbs:identifiedBy ?identifier_assertion .
                    ?identifier_assertion cbs:hasValue ?identifier_value ;
                                         cbs:derivedBy ?identifier_source .
                    BIND(REPLACE(str(?identifier_source), ".*[/#]", "") AS ?identifier_source_name)
                }}
            }}
            GROUP BY ?smiles ?primary_name ?synonym
        """
        results = await self.query(sparql_query)
        if not results:
            return None

        # Parse all_identifiers into a structured format
        all_identifiers = {}
        if results and results[0].get("?all_identifiers"):
            for identifier in results[0]["?all_identifiers"].split("|"):
                if ":" in identifier:
                    source, value = identifier.split(":", 1)
                    if source not in all_identifiers:
                        all_identifiers[source] = []
                    all_identifiers[source].append(value)

        return {
            "id": compound_id,
            "smiles": results[0].get("?smiles", ""),
            "name": results[0].get("?primary_name", ""),
            "synonyms": list({r.get("?synonym") for r in results if r.get("?synonym")}),
            "identifiers": all_identifiers,
        }

    async def get_disease_info(self, disease_id: str) -> dict[str, Any] | None:
        """Get comprehensive disease information by disease ID (Mondo ID or MeSH ID)."""
        # Handle different ID formats
        clean_id = disease_id
        is_mesh_input = False
        is_mondo_input = False

        if disease_id.upper().startswith("MONDO"):
            # Extract just the numeric part from MONDO:0019737 or MONDO_0019737
            clean_id = disease_id.replace(":", "").replace("_", "")[5:]
            is_mondo_input = True
        elif disease_id.upper().startswith("MESH"):
            # Handle MeSH IDs like MESH:D019737 or MESH_D019737
            clean_id = disease_id.replace(":", "").replace("_", "")[4:]
            is_mesh_input = True
        elif disease_id.startswith("D") and len(disease_id) > 1 and disease_id[1:].isdigit():
            is_mesh_input = True

        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            
            SELECT DISTINCT ?name ?definition ?mesh_id ?umls_id ?ncit_id ?orphanet_id
                (GROUP_CONCAT(DISTINCT ?synonym; separator=" | ") as ?synonyms)
                (GROUP_CONCAT(DISTINCT ?broader_name; separator=" | ") as ?broader_diseases)
                (GROUP_CONCAT(DISTINCT ?narrower_name; separator=" | ") as ?narrower_diseases)
                (GROUP_CONCAT(DISTINCT CONCAT(?standard_name, ":", ?id_value); separator=" | ") as ?external_ids)
            WHERE {{
                {{
                    # Try mondo_ prefixed ID (for MONDO input)
                    BIND(cbs:mondo_{clean_id} AS ?disease)
                }} UNION {{
                    # Try direct ID lookup (for raw numeric)
                    ?disease a cbs:Disease ;
                            cbs:hasId "{clean_id}" .
                }} UNION {{
                    # Try lookup by MeSH ID using identifiedBy pattern
                    ?disease a cbs:Disease ;
                            cbs:identifiedBy ?mesh_lookup .
                    ?mesh_lookup cbs:hasValue "{clean_id}" ;
                                cbs:derivedBy cbs:standard_mesh .
                }} UNION {{
                    # Alternative MeSH lookup using assertedBy pattern (similar to external_ids)
                    ?disease a cbs:Disease ;
                            cbs:identifiedBy ?mesh_lookup2 .
                    ?mesh_lookup2 cbs:hasValue "{clean_id}" ;
                                cbs:assertedBy ?mesh_standard .
                    FILTER(CONTAINS(str(?mesh_standard), "mesh"))
                }}

                # Basic properties
                OPTIONAL {{ ?disease cbs:hasName ?name }}
                OPTIONAL {{ ?disease cbs:hasDefinition ?definition }}

                # Synonyms
                OPTIONAL {{
                    ?disease cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym .
                }}

                # Specific identifier lookups
                OPTIONAL {{
                    ?disease cbs:identifiedBy ?mesh_identifier .
                    ?mesh_identifier cbs:derivedBy cbs:standard_mesh ;
                                    cbs:hasValue ?mesh_id .
                }}

                OPTIONAL {{
                    ?disease cbs:identifiedBy ?umls_identifier .
                    ?umls_identifier cbs:derivedBy cbs:standard_umls ;
                                    cbs:hasValue ?umls_id .
                }}

                OPTIONAL {{
                    ?disease cbs:identifiedBy ?ncit_identifier .
                    ?ncit_identifier cbs:derivedBy cbs:standard_ncit ;
                                    cbs:hasValue ?ncit_id .
                }}

                OPTIONAL {{
                    ?disease cbs:identifiedBy ?orphanet_identifier .
                    ?orphanet_identifier cbs:derivedBy cbs:standard_orphanet ;
                                        cbs:hasValue ?orphanet_id .
                }}

                # External identifiers (general) - using both patterns
                OPTIONAL {{
                    ?disease cbs:identifiedBy ?id .
                    {{
                        ?id cbs:assertedBy ?standard ;
                            cbs:hasValue ?id_value .
                        BIND(REPLACE(str(?standard), ".*[/#]", "") AS ?standard_name)
                    }} UNION {{
                        ?id cbs:derivedBy ?standard2 ;
                            cbs:hasValue ?id_value .
                        BIND(REPLACE(str(?standard2), ".*[/#]", "") AS ?standard_name)
                    }}
                }}

                # Broader diseases (parent categories)
                OPTIONAL {{
                    ?disease skos:broader ?broader .
                    ?broader cbs:hasName ?broader_name .
                }}

                # Narrower diseases (subcategories)
                OPTIONAL {{
                    ?disease skos:narrower ?narrower .
                    ?narrower cbs:hasName ?narrower_name .
                }}
            }}
            GROUP BY ?name ?definition ?mesh_id ?umls_id ?ncit_id ?orphanet_id
        """

        results = await self.query(sparql_query)
        if not results:
            return None

        # Find the first non-empty result
        selected_result = None
        for result in results:
            # Check if this result has actual data (not all empty)
            if result.get("?name") or result.get("?definition") or result.get("?mesh_id") or result.get("?synonyms"):
                selected_result = result
                break

        if not selected_result:
            return None

        # Parse external_ids into structured format
        external_identifiers = {}
        if selected_result.get("?external_ids"):
            for identifier in selected_result["?external_ids"].split(" | "):
                if ":" in identifier:
                    source, value = identifier.split(":", 1)
                    if source not in external_identifiers:
                        external_identifiers[source] = []
                    external_identifiers[source].append(value)

        # Extract MONDO ID from external identifiers if available
        mondo_id = ""
        if is_mondo_input:
            mondo_id = f"MONDO:{clean_id}"
        elif "standard_mondo" in external_identifiers:
            # Extract MONDO ID from external identifiers
            mondo_ids = external_identifiers["standard_mondo"]
            if mondo_ids:
                mondo_id = f"MONDO:{mondo_ids[0]}"

        # Parse synonyms
        synonyms = []
        if selected_result.get("?synonyms"):
            synonyms = [s.strip() for s in selected_result["?synonyms"].split(" | ") if s.strip()]

        # Parse broader diseases
        broader_diseases = []
        if selected_result.get("?broader_diseases"):
            broader_diseases = [s.strip() for s in selected_result["?broader_diseases"].split(" | ") if s.strip()]

        # Parse narrower diseases
        narrower_diseases = []
        if selected_result.get("?narrower_diseases"):
            narrower_diseases = [s.strip() for s in selected_result["?narrower_diseases"].split(" | ") if s.strip()]

        return {
            "id": disease_id,
            "mondo_id": mondo_id,
            "mesh_id": selected_result.get("?mesh_id", "") or (clean_id if is_mesh_input else ""),
            "umls_id": selected_result.get("?umls_id", ""),
            "ncit_id": selected_result.get("?ncit_id", ""),
            "orphanet_id": selected_result.get("?orphanet_id", ""),
            "name": selected_result.get("?name", ""),
            "definition": selected_result.get("?definition", ""),
            "synonyms": synonyms,
            "broader_diseases": broader_diseases,  # Parent disease categories
            "narrower_diseases": narrower_diseases,  # Child disease subcategories
            "external_identifiers": external_identifiers,  # Structured by source
        }

    async def get_cellular_context_info(self, name_filter: str = None) -> dict[str, Any] | None:
        """Get cellular context information (cell lines and cell types) by name."""

        # Build filter clause
        filter_clause = ""
        if name_filter:
            filter_clause = f'FILTER(CONTAINS(LCASE(?name), "{name_filter.lower()}"))'

        sparql_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT DISTINCT ?name
                   ?id
                   ?cell_type
                   ?description
                   (GROUP_CONCAT(DISTINCT ?synonym_value; separator=" | ") as ?synonyms)
                   ?sex
                   ?origin_type
                   ?derived_tissue_name
                   ?native_cell_name
                   ?native_cell_description
                   (COUNT(DISTINCT ?pubmed_ref) as ?publication_count)
            WHERE {{
                # Find by name (works for both cell types and lines)
                ?cell a cbs:CellLine ;
                      cbs:hasName ?name ;
                      cbs:hasId ?id .
                
                {filter_clause}

                # Get cell type (CulturedCell or NativeCell)
                OPTIONAL {{
                    ?cell a ?specific_type .
                    FILTER(?specific_type IN (cbs:CulturedCell, cbs:NativeCell))
                    BIND(STRAFTER(STR(?specific_type), "#") AS ?cell_type)
                }}

                # Basic properties
                OPTIONAL {{ ?cell cbs:hasDescription ?description }}
                OPTIONAL {{ ?cell cbs:associatesWithSex ?sex }}
                OPTIONAL {{
                    ?cell cbs:hasOrigin ?origin .
                    BIND(STRAFTER(STR(?origin), "cell_line_") AS ?origin_type)
                }}

                # Synonyms
                OPTIONAL {{
                    ?cell cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synonym_value .
                }}

                # Tissue and native cell information
                OPTIONAL {{
                    ?cell cbs:derivedFromTissue ?tissue .
                    ?tissue cbs:hasName ?derived_tissue_name .
                }}
                OPTIONAL {{
                    ?cell cbs:derivedFromNativeCell ?native_cell .
                    ?native_cell cbs:hasName ?native_cell_name .
                    OPTIONAL {{ ?native_cell cbs:hasDescription ?native_cell_description }}
                }}

                # Literature references
                OPTIONAL {{
                    ?mention cbs:refersToCellLine ?cell ;
                            cbs:cellLineMentionContainedBy ?pubmed_ref .
                }}
            }}
            GROUP BY ?name ?id ?cell_type ?description ?sex ?origin_type
                     ?derived_tissue_name ?native_cell_name ?native_cell_description
        """

        results = await self.query(sparql_query)
        if not results:
            return None

        # Process results - can return multiple cell lines
        processed_results = []
        for result in results:
            # Parse synonyms
            synonyms = []
            if result.get("?synonyms"):
                synonyms = [s.strip() for s in result["?synonyms"].split(" | ") if s.strip()]

            # Convert publication count to integer
            pub_count = 0
            if result.get("?publication_count"):
                try:
                    pub_count = int(result.get("?publication_count"))
                except (ValueError, TypeError):
                    pub_count = 0

            cell_info = {
                "id": result.get("?id", ""),
                "name": result.get("?name", ""),
                "cell_type": result.get("?cell_type", ""),
                "description": result.get("?description", ""),
                "synonyms": synonyms,
                "sex": result.get("?sex", ""),
                "origin_type": result.get("?origin_type", ""),
                "derived_tissue": result.get("?derived_tissue_name", ""),
                "native_cell_name": result.get("?native_cell_name", ""),
                "native_cell_description": result.get("?native_cell_description", ""),
                "publication_count": pub_count,
            }
            processed_results.append(cell_info)

        # If only one result, return it directly
        if len(processed_results) == 1:
            return processed_results[0]

        # Multiple results - return structured response
        return {"results": processed_results, "count": len(processed_results)}
