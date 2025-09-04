import asyncio
from typing import Any, Literal, Union

import pandas as pd
import requests
from loguru import logger
from pydantic import BaseModel, Field

from explain.utils.bq import retrieve_from_bigquery
from explain.utils.chem.chembl import ChEMBLClient
from explain.utils.chem.mol_utils import to_inchikey
from explain.utils.chem.pubchem import PubChemClient
from explain.utils.kg_client import KGClient
from explain.utils.uniprot import UniProtMapper


class BaseEntity(BaseModel):
    """Base class for biological entities."""

    id: str | None = None
    name: str | None = None
    synonyms: list[str] = Field(default_factory=list)

    # KG client instance
    _kg_client: KGClient | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._kg_client = KGClient()

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return self.model_dump(exclude={"_kg_client"})

    async def close(self):
        """Close KG client connection."""
        if self._kg_client:
            await self._kg_client.close()

    def __eq__(self, other: "BaseEntity") -> bool:
        """Compare two entities by their IDs."""
        if self.id and other.id:
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        """Hash the entity by its ID."""
        return hash(self.id)

    @classmethod
    async def from_id(cls, id: str, **kwargs):
        """Create a base entity from an ID."""
        out = cls(id=id)
        await out.update(**kwargs)
        return out

    async def update(self, **kwargs) -> "BaseEntity":
        """
        Update entity with information from KG and other sources.
        This is the main method to populate entity data.

        Returns:
            self: Updated entity instance
        """
        raise NotImplementedError("Subclasses must implement update method")


class GeneEntity(BaseEntity):
    """Gene entity with Ensembl ID as primary identifier."""

    # Primary identifier
    id: str  # Ensembl ID

    # Names and symbols
    symbol: str | None = None  # gene approved symbol
    name: str | None = None  # gene name
    synonyms: list[str] = Field(default_factory=list)

    # External identifiers - all using snake_case
    entrez_id: str | None = None
    refseq_id: str | None = None
    uniprot_id: str | None = None
    uniprot_name: str | None = None
    hgnc_id: str | None = None
    chembl_id: str | None = None
    hpa_id: str | None = None
    uniprot_function_summary: str | None = None
    perturbation_type: Literal["knockout", "overexpression", "knockdown"] = "knockout"

    @classmethod
    async def from_name(cls, name: str) -> "GeneEntity":
        """Create a gene entity from a name."""
        kg_client = KGClient()
        name_query = f"""
        PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
        SELECT ?id ?name
        WHERE {{
            ?gene cbs:hasApprovedSymbol "{name}" ;
                cbs:hasId ?id .
            BIND("{name}" AS ?name)
        }}
        """
        out = await kg_client.query(name_query)
        tmp_df = pd.DataFrame(out)
        tmp_df.columns = tmp_df.columns.map(lambda x: x.strip("?"))
        if not tmp_df.empty:
            return await cls.from_id(tmp_df.iloc[0]["id"])
        raise ValueError(f"No gene found for name: {name}")

    async def update(self, kg_only: bool = False) -> "GeneEntity":
        """
        Update gene entity with information from KG and UniProt.

        Returns:
            self: Updated gene entity
        """
        if not self.id:
            raise ValueError("Gene ID (Ensembl ID) is required for update")

        # First, try to get information from KG
        if self._kg_client:
            try:
                kg_info = await self._kg_client.get_gene_info(self.id)
                if kg_info:
                    self._update_from_kg(kg_info)
            except Exception as e:
                logger.warning(f"Failed to get gene info from KG: {e}")

        # Fallback to UniProt mapping if we have Entrez ID
        if not kg_only or (self.entrez_id and not self.uniprot_id):
            if self.entrez_id is not None:
                query_id = self.entrez_id
                from_db = "GeneID"
            else:
                query_id = self.id
                from_db = "Ensembl"
            uniprot_info = await UniProtMapper.map(query_id, from_db=from_db, tax_id=9606)
            self._update_from_uniprot(uniprot_info)

        return self

    def _update_from_uniprot(self, uniprot_info: dict[str, Any]):
        """Update entity from UniProt information."""
        self.uniprot_id = uniprot_info.get("uniprot_id") or self.uniprot_id
        self.uniprot_name = uniprot_info.get("uniprot_name") or self.uniprot_name
        self.hgnc_id = uniprot_info.get("hgnc_id") or self.hgnc_id
        self.chembl_id = uniprot_info.get("chembl_id") or self.chembl_id
        self.hpa_id = uniprot_info.get("hpa_id") or self.hpa_id
        self.uniprot_function_summary = uniprot_info.get("uniprot_function_summary") or self.uniprot_function_summary

    async def get_summary(self) -> str:
        """Get the summary function of the gene."""
        if self.uniprot_function_summary is None:
            uniprot_info = asyncio.run(UniProtMapper.map(self.id, from_db="Ensembl", tax_id=9606))
            self._update_from_uniprot(uniprot_info)
        return self.uniprot_function_summary

    async def get_annotations(self) -> pd.DataFrame:
        """
        Get detailed annotations for the gene including molecular functions, biological processes,
        disease associations, pathways, and other key information.

        Returns:
            DataFrame with columns: category, annotation_count, annotations
        """
        if not self.symbol:
            raise ValueError("Gene symbol is required to get annotations")

        if not self._kg_client:
            raise ValueError("KG client is not initialized")

        sparql_query = f"""
        PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?category (COUNT(DISTINCT ?annotation) as ?count) (GROUP_CONCAT(DISTINCT ?annotation; separator=" | ") as ?annotations)
        WHERE {{
            {{
                # Molecular Functions (excluding regulation/process terms)
                SELECT ("Molecular Functions" as ?category) (STR(?funcName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:encodesProtein ?protein .
                    ?protein cbs:hasFunction ?function .
                    ?function rdfs:label ?funcName .
                    FILTER(!CONTAINS(LCASE(STR(?funcName)), "regulation") &&
                           !CONTAINS(LCASE(STR(?funcName)), "process"))
                }}
            }} UNION {{
                # Biological Processes
                SELECT ("Biological Processes" as ?category) (STR(?procName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:encodesProtein ?protein .
                    ?protein cbs:hasFunction ?process .
                    ?process rdfs:label ?procName .
                    FILTER(CONTAINS(LCASE(STR(?procName)), "regulation") ||
                           CONTAINS(LCASE(STR(?procName)), "process"))
                }}
            }} UNION {{
                # Disease Associations
                SELECT ("Disease Associations" as ?category) (STR(?diseaseName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" .
                    {{
                        ?assoc cbs:hasGeneAssociate ?gene ;
                               cbs:associatesWithDisease ?disease .
                    }} UNION {{
                        ?gene cbs:associatesWithDisease ?disease .
                    }}
                    ?disease cbs:hasName ?diseaseName .
                }}
            }} UNION {{
                # Key Signaling Pathways
                SELECT ("Signaling Pathways" as ?category) (STR(?pathwayName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:encodesProtein ?protein .
                    ?pathway cbs:containsProtein ?protein ;
                             cbs:hasName ?pathwayName .
                    FILTER(CONTAINS(LCASE(STR(?pathwayName)), "signal") ||
                           CONTAINS(LCASE(STR(?pathwayName)), "pathway"))
                }}
            }} UNION {{
                # Gene Groups/Families
                SELECT ("Gene Groups" as ?category) (STR(?groupName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:isMemberOf ?group .
                    ?group cbs:hasName ?groupName .
                }}
            }} UNION {{
                # Expression Patterns
                SELECT ("Tissue Expression" as ?category) (STR(?tissueName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:encodesProtein ?protein .
                    ?protein cbs:proteinExpressedIn ?tissue .
                    ?tissue cbs:hasName ?tissueName .
                }}
            }} UNION {{
                # Protein Domains
                SELECT ("Protein Domains" as ?category) (STR(?domainName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:encodesProtein ?protein .
                    ?protein cbs:containsDomain ?domain .
                    ?domain cbs:hasName ?domainName .
                }}
            }} UNION {{
                # Synonyms
                SELECT ("Gene Synonyms" as ?category) (STR(?synName) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:hasSynonym ?syn .
                    ?syn cbs:hasValue ?synName .
                }}
            }} UNION {{
                # Cellular Location
                SELECT ("Cellular Location" as ?category) (STR(?location) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:encodesProtein ?protein .
                    {{
                        ?protein cbs:proteinLocalizedTo ?loc .
                        ?loc rdfs:label ?location .
                    }} UNION {{
                        ?protein cbs:hasFunction ?func .
                        ?func rdfs:label ?location .
                        FILTER(CONTAINS(LCASE(STR(?location)), "membrane") ||
                               CONTAINS(LCASE(STR(?location)), "cytoplasm") ||
                               CONTAINS(LCASE(STR(?location)), "cellular"))
                    }}
                }}
            }} UNION {{
                # External IDs
                SELECT ("External Identifiers" as ?category) (CONCAT(?idSource, ": ", STR(?idValue)) as ?annotation)
                WHERE {{
                    ?gene cbs:hasApprovedSymbol "{self.symbol}" ;
                          cbs:identifiedBy ?id .
                    ?id cbs:assertedBy ?idSource ;
                        cbs:hasValue ?idValue .
                }}
            }}
        }}
        GROUP BY ?category
        ORDER BY ?category
        """

        try:
            results = await self._kg_client.query(sparql_query)

            if not results:
                logger.warning(f"No annotations found for gene: {self.symbol}")
                return pd.DataFrame(columns=["category", "annotation_count", "annotations"])

            # Convert to DataFrame and clean up column names
            df = pd.DataFrame(results)

            # Rename columns to remove the "?" prefix and make them cleaner
            column_mapping = {"?category": "category", "?count": "annotation_count", "?annotations": "annotations"}

            df = df.rename(columns=column_mapping)

            # Convert count column to integer if it exists
            if "annotation_count" in df.columns:
                df["annotation_count"] = pd.to_numeric(df["annotation_count"], errors="coerce").fillna(0).astype(int)

            # Sort by category for consistent output
            df = df.sort_values("category").reset_index(drop=True)

            logger.debug(f"Retrieved {len(df)} annotation categories for gene: {self.symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to get annotations for gene {self.symbol}: {e}")
            return pd.DataFrame(columns=["category", "annotation_count", "annotations"])

    def _update_from_kg(self, kg_info: dict[str, Any]):
        """Update entity from KG information."""
        self.symbol = kg_info.get("symbol") or self.symbol
        self.name = kg_info.get("name") or self.name
        self.entrez_id = kg_info.get("entrez_id") or self.entrez_id
        self.refseq_id = kg_info.get("refseq_id") or self.refseq_id
        self.uniprot_id = kg_info.get("uniprot_id") or self.uniprot_id
        self.hgnc_id = kg_info.get("hgnc_id") or self.hgnc_id
        self.synonyms = kg_info.get("synonyms", []) or self.synonyms

    async def _uniprot_mapping(self, tax_id: int = 9606):
        """
        Map gene identifiers using UniProt API.

        Args:
            tax_id (int): Taxonomy ID (default: 9606 for human).
        """

        submit_url = "https://rest.uniprot.org/idmapping/run"
        submission = requests.post(
            submit_url,
            data={"from": "GeneID", "to": "UniProtKB-Swiss-Prot", "ids": self.entrez_id, "taxId": tax_id},
        )
        job_id = submission.json().get("jobId")
        if not job_id:
            logger.error("Failed to submit UniProt mapping job.")
            return

        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        while True:
            status_res = requests.get(status_url).json()
            if status_res.get("jobStatus") == "COMPLETE" or "results" in status_res:
                break
            await asyncio.sleep(1)

        results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
        results = requests.get(results_url).json().get("results", [])

        if results:
            self.uniprot_id = results[0].get("to")
            entry_url = f"https://rest.uniprot.org/uniprotkb/{self.uniprot_id}"
            entry_data = requests.get(entry_url).json()
            self.uniprot_id = entry_data.get("uniProtkbId")

            for row in entry_data.get("uniProtKBCrossReferences", []):
                db = row.get("database")
                if db == "ChEMBL":
                    self.chembl_id = row.get("id")
                elif db == "HPA":
                    self.hpa_id = row.get("id")

    async def get_process_interaction(self, other_gene: Union[str, "GeneEntity"]) -> list[dict[str, Any]]:
        """
        Get interaction types between this gene and another gene.

        Args:
            other_gene: Either a gene symbol (str) or GeneEntity object

        Returns:
            List of interaction dictionaries with relationship_type, intermediate, and intermediate_name
        """
        if not self.symbol:
            raise ValueError("This gene must have a symbol to query interactions")

        # Get the other gene's symbol
        if isinstance(other_gene, str):
            other_symbol = other_gene
        elif isinstance(other_gene, GeneEntity):
            if not other_gene.symbol:
                raise ValueError("Other gene must have a symbol to query interactions")
            other_symbol = other_gene.symbol
        else:
            raise ValueError("other_gene must be either a string (gene symbol) or GeneEntity")

        if not self._kg_client:
            raise ValueError("KG client is not initialized")

        sparql_query = f"""
        PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?relationship_type ?intermediate ?intermediate_name
        WHERE {{
            ?gene1 cbs:hasApprovedSymbol "{self.symbol}" .
            ?gene2 cbs:hasApprovedSymbol "{other_symbol}" .

            {{
                # Direct relationships
                ?gene1 ?relationship_type ?gene2 .
                BIND("direct_relationship" AS ?intermediate)
                BIND("" AS ?intermediate_name)
            }}
            UNION
            {{
                # Protein-protein relationships
                ?gene1 cbs:encodesProtein ?protein1 .
                ?gene2 cbs:encodesProtein ?protein2 .
                ?protein1 ?relationship_type ?protein2 .
                BIND("protein_interaction" AS ?intermediate)
                BIND(str(?protein1) AS ?intermediate_name)
            }}
            UNION
            {{
                # Shared pathways
                ?gene1 cbs:encodesProtein ?protein1 .
                ?gene2 cbs:encodesProtein ?protein2 .
                ?pathway cbs:containsProtein ?protein1, ?protein2 ;
                        cbs:hasName ?pathway_name .
                BIND(cbs:containsProtein AS ?relationship_type)
                BIND("shared_pathway" AS ?intermediate)
                BIND(?pathway_name AS ?intermediate_name)
            }}
            UNION
            {{
                # Shared functions
                ?gene1 cbs:encodesProtein ?protein1 .
                ?gene2 cbs:encodesProtein ?protein2 .
                ?protein1 cbs:hasFunction ?function .
                ?protein2 cbs:hasFunction ?function .
                ?function rdfs:label ?function_name .
                BIND(cbs:hasFunction AS ?relationship_type)
                BIND("shared_function" AS ?intermediate)
                BIND(?function_name AS ?intermediate_name)
            }}
        }}
        """

        try:
            results = await self._kg_client.query(sparql_query)

            # Process results into structured format
            interactions = []
            for result in results:
                interaction = {
                    "relationship_type": result.get("?intermediate"),
                    "relationship": result.get("?intermediate_name"),
                    "gene1_symbol": self.symbol,
                    "gene2_symbol": other_symbol,
                }
                interactions.append(interaction)

            logger.debug(f"Found {len(interactions)} interactions between {self.symbol} and {other_symbol}")
            return interactions

        except Exception as e:
            logger.error(f"Failed to get interactions between {self.symbol} and {other_symbol}: {e}")
            return []


class CompoundEntity(BaseEntity):
    """Compound entity with InChIKey as primary identifier."""

    # Primary identifiers
    id: str | None = None  # InChIKey
    chembl_id: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    # Chemical properties
    smiles: str | None = None
    inchikey: str | None = None
    # External identifiers
    pubchem_cid: str | None = None
    rec_id: str | None = None
    dose: str | None = None

    @classmethod
    async def from_name(cls, name: str, **kwargs) -> "CompoundEntity":
        """Create a compound entity from an ID."""
        # try to find a compound by its name
        pubchem_client = PubChemClient()
        pubchem_info = pubchem_client.get_compound_info(name=name)

        data = cls(
            id=pubchem_info.get("inchikey"),
            inchikey=pubchem_info.get("inchikey"),
            synonyms=pubchem_info.get("synonyms", []),
            chembl_id=pubchem_info.get("chembl_id"),
        )
        return await data.update(**kwargs)

    async def update(self, kg_only: bool = False, **kwargs) -> "CompoundEntity":
        """
        Update compound entity with information from KG and other sources.

        Returns:
            self: Updated compound entity
        """
        # Determine primary identifier
        if self.id.startswith("CHEMBL"):
            self.chembl_id = self.id
            self.id = None

        if not self.id and not self.chembl_id:
            raise ValueError("Either InChIKey (id) or ChEMBL ID is required for update")

        # Set InChIKey as primary ID if we have it
        if self.inchikey and not self.id:
            self.id = self.inchikey

        compound_info = None
        # primary identifier should always be inchikey
        if self.chembl_id:
            compound_info = await self._kg_client.get_compound_info(compound_id=self.chembl_id)
            self._update_from_kg(compound_info)
        else:
            # we start with a chembl query to get the information about the molecule
            # if that fails, we fallback to pubchem
            # finally we add the information from the KG and rec_id if possible
            chembl_client = ChEMBLClient()
            compound_info = chembl_client.get_compound_info(inchikey=self.id)
            self._update_from_chembl(compound_info)

        if not self.smiles:
            pubchem_client = PubChemClient()
            compound_info = pubchem_client.get_compound_info_by_inchikey(self.inchikey)
            self._update_from_pubchem(compound_info)

        if self.inchikey is None and self.smiles:
            self.inchikey = to_inchikey(self.smiles)

        self.id = self.inchikey
        await self._fetch_rxrx_id()

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def _update_from_pubchem(self, pubchem_info: dict[str, Any]):
        """Update entity from PubChem information."""
        self.smiles = self.smiles or pubchem_info.get("smiles")
        self.name = self.name or pubchem_info.get("name")
        self.synonyms = self.synonyms or pubchem_info.get("synonyms", [])
        self.chembl_id = self.chembl_id or pubchem_info.get("chembl_id")
        self.inchikey = self.inchikey or pubchem_info.get("inchikey")
        self.pubchem_cid = self.pubchem_cid or pubchem_info.get("cid")

    def _update_from_chembl(self, chembl_info: dict[str, Any]):
        """Update entity from ChEMBL information."""
        self.smiles = self.smiles or chembl_info.get("smiles")
        self.name = self.name or chembl_info.get("name")
        self.synonyms = self.synonyms or chembl_info.get("synonyms", [])
        self.chembl_id = self.chembl_id or chembl_info.get("chembl_id")
        self.inchikey = self.inchikey or chembl_info.get("inchikey")

    def _update_from_kg(self, kg_info: dict[str, Any]):
        """Update entity from KG information."""
        self.smiles = self.smiles or kg_info.get("smiles")
        self.name = self.name or kg_info.get("name")
        self.synonyms = self.synonyms or kg_info.get("synonyms", [])
        self.chembl_id = self.chembl_id or kg_info.get("id")
        # Parse identifiers
        identifiers = kg_info.get("identifiers", {})
        if "standard_pubchem" in identifiers:
            self.pubchem_cid = identifiers["standard_pubchem"][0]

    async def _fetch_rxrx_id(self):
        """Fetch REC_ID for the compound from the RXRX datalake."""
        if not self.inchikey:
            return

        sql = f"""
            SELECT rec_id
            FROM cauldron__cauldron.compounds
            WHERE inchi_key = '{self.inchikey}'
        """
        res = retrieve_from_bigquery(sql)

        if len(res) == 1:
            self.rec_id = res.loc[0, "rec_id"]


class DiseaseEntity(BaseEntity):
    """Disease entity with MONDO ID as primary identifier."""

    # Primary identifier
    id: str | None = None  # MONDO ID
    name: str | None = None
    # Alternative identifiers
    mondo_id: str | None = None
    mesh_id: str | None = None
    umls_id: str | None = None
    ncit_id: str | None = None
    orphanet_id: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    # Disease information
    definition: str | None = None
    broader_diseases: list[str] = Field(default_factory=list)
    narrower_diseases: list[str] = Field(default_factory=list)

    async def update(self) -> "DiseaseEntity":
        """
        Update disease entity with information from KG.

        Returns:
            self: Updated disease entity
        """
        if not self.id and not self.mondo_id and not self.mesh_id:
            raise ValueError("Disease ID (MONDO ID or MeSH ID) is required for update")

        # Set primary ID
        if not self.id:
            if self.mondo_id:
                self.id = self.mondo_id
            elif self.mesh_id:
                self.id = self.mesh_id

        # Get information from KG
        if self._kg_client:
            try:
                kg_info = await self._kg_client.get_disease_info(self.id)
                if kg_info:
                    self._update_from_kg(kg_info)
            except Exception as e:
                logger.warning(f"Failed to get disease info from KG: {e}")

        return self

    def _update_from_kg(self, kg_info: dict[str, Any]):
        """Update entity from KG information."""
        self.mondo_id = kg_info.get("mondo_id") or self.mondo_id
        self.mesh_id = kg_info.get("mesh_id") or self.mesh_id
        self.umls_id = kg_info.get("umls_id") or self.umls_id
        self.ncit_id = kg_info.get("ncit_id") or self.ncit_id
        self.orphanet_id = kg_info.get("orphanet_id") or self.orphanet_id
        self.name = kg_info.get("name") or self.name
        self.definition = kg_info.get("definition") or self.definition
        self.synonyms = kg_info.get("synonyms", []) or self.synonyms
        self.broader_diseases = kg_info.get("broader_diseases", []) or self.broader_diseases
        self.narrower_diseases = kg_info.get("narrower_diseases", []) or self.narrower_diseases
        # always prefer mondo_id over mesh_id as primary id
        self.id = self.mondo_id or self.mesh_id


class CellContext(BaseEntity):
    """Cell type entity with ontology ID as primary identifier."""

    # Primary identifier
    id: str | None = None  # Ontology ID
    cell_type: str | None = None
    culture_type: str | None = None  # whether primary or cell line
    description: str | None = None
    sex: str | None = None
    origin_type: str | None = None
    derived_tissue: str | None = None
    name: str | None = None

    @classmethod
    async def from_name(cls, name: str, **kwargs) -> "CellContext":
        """Create a cell context entity from a name."""
        cell_context = cls(name=name, **kwargs)
        await cell_context.update()
        return cell_context

    async def update(self) -> "CellContext":
        """
        Update cell type entity with information from KG.

        Returns:
            self: Updated cell type entity
        """
        if not self.id and not self.name:
            raise ValueError("Cell type ID or name is required for update")

        # Get information from KG
        if self._kg_client:
            try:
                kg_info = await self._kg_client.get_cellular_context_info(self.name)
                if kg_info:
                    if isinstance(kg_info, dict) and "results" in kg_info:
                        # Multiple results - use the first one
                        if kg_info["results"]:
                            self._update_from_kg(kg_info["results"][0])
                    else:
                        self._update_from_kg(kg_info)
            except Exception as e:
                logger.warning(f"Failed to get cell type info from KG: {e}")
        return self

    def _update_from_kg(self, kg_info: dict[str, Any]):
        """Update entity from KG information."""
        self.id = kg_info.get("id") or self.id
        self.name = kg_info.get("name") or self.name
        self.cell_type = kg_info.get("cell_type") or self.cell_type
        self.description = kg_info.get("description") or self.description
        self.synonyms = kg_info.get("synonyms", []) or self.synonyms
        self.sex = kg_info.get("sex") or self.sex
        self.origin_type = kg_info.get("origin_type") or self.origin_type
        self.derived_tissue = kg_info.get("derived_tissue") or self.derived_tissue


class PerturbationEntity(BaseEntity):
    """Perturbation entity representing experimental perturbations."""

    # Core perturbation components
    genes: list[GeneEntity] = Field(default_factory=list, description="List of gene entities")
    compounds: list[CompoundEntity] = Field(default_factory=list, description="List of compound entities")
    cell_context: CellContext | None = Field(default=None, description="Cell context entity")
    disease: DiseaseEntity | None = Field(default=None, description="Disease entity if applicable")
    description: str | None = Field(default=None, description="Human-readable description of the perturbation")

    def add_gene(self, gene: GeneEntity):
        """Add a gene to the perturbation."""
        if gene not in self.genes:
            self.genes.append(gene)

    def add_compound(self, compound: CompoundEntity):
        """Add a compound to the perturbation."""
        if compound not in self.compounds:
            self.compounds.append(compound)

    def __repr__(self) -> str:
        """Detailed string representation of the perturbation."""
        components = []
        if self.genes:
            components.append(f"genes={len(self.genes)}")
        if self.compounds:
            components.append(f"compounds={len(self.compounds)}")
        if self.cell_context:
            components.append(f"cell_context={self.cell_context.name}")
        if self.disease:
            components.append(f"disease={self.disease.name}")

        return f"PerturbationEntity({', '.join(components)})"
