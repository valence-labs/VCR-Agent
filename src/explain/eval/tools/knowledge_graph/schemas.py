"""Pydantic schemas for knowledge graph tool arguments."""

from typing import Literal

from pydantic import BaseModel, Field


class EntityFactsArgs(BaseModel):
    """Arguments for retrieving facts about biomedical entities."""

    entity_id: str = Field(description="Entity identifier (e.g., ENSEMBL:ENSG00000139618 for genes)")
    entity_type: Literal["GENE", "DISEASE", "CHEMICAL"] = Field(description="Type of entity to query")
    entity_label: str | None = Field(default=None, description="Optional human-readable label for the entity")


class EntityRelationshipArgs(BaseModel):
    """Arguments for finding relationships between entities."""

    source_entity_id: str = Field(description="Source entity identifier")
    source_entity_type: Literal["GENE", "DISEASE", "CHEMICAL"] = Field(description="Type of source entity")
    target_entity_id: str | None = Field(default=None, description="Target entity identifier (if specified)")
    target_entity_type: Literal["GENE", "DISEASE", "CHEMICAL"] | None = Field(
        default=None, description="Type of target entity"
    )
    relationship_types: list[str] | None = Field(
        default=None, description="Specific relationship types to search for"
    )


class PublicationQueryArgs(BaseModel):
    """Arguments for querying publication information."""

    pubmed_ids: list[str] = Field(description="List of PubMed IDs to get information for")
    include_citations: bool = Field(default=True, description="Whether to include citation count")
    include_journal: bool = Field(default=True, description="Whether to include journal information")


class SparqlQueryArgs(BaseModel):
    """Arguments for executing custom SPARQL queries."""

    query: str = Field(description="SPARQL query to execute against the knowledge graph")
    limit: int | None = Field(default=100, description="Maximum number of results to return")


class PerturbationQueryArgs(BaseModel):
    """Arguments for querying perturbation effects and activities."""

    perturbation_id: str = Field(description="Perturbation identifier (compound ID, gene ID for knockout, etc.)")
    perturbation_type: Literal["COMPOUND", "GENE_KNOCKOUT", "OVEREXPRESSION", "INTERVENTION"] = Field(
        description="Type of perturbation"
    )
    target_entity_id: str | None = Field(default=None, description="Optional target entity to focus on")
    include_activities: bool = Field(default=True, description="Include activity measurements")
    include_assays: bool = Field(default=True, description="Include assay results")
    include_phenotypes: bool = Field(default=True, description="Include phenotypic outcomes")


class CellularContextArgs(BaseModel):
    """Arguments for querying cellular context and environment."""

    entity_id: str = Field(description="Gene, protein, or other entity ID to query context for")
    cell_line_id: str | None = Field(default=None, description="Specific cell line context")
    include_expression: bool = Field(default=True, description="Include expression patterns")
    include_localization: bool = Field(default=True, description="Include cellular localization")
    include_processes: bool = Field(default=True, description="Include biological processes")
    include_pathways: bool = Field(default=True, description="Include pathway information")


class PhenotypeAssociationArgs(BaseModel):
    """Arguments for querying phenotype and disease associations."""

    entity_id: str = Field(description="Entity ID (gene, compound, etc.) to find phenotypes for")
    entity_type: Literal["GENE", "COMPOUND", "ACTIVITY", "ASSAY"] = Field(description="Type of entity")
    include_diseases: bool = Field(default=True, description="Include disease associations")
    include_traits: bool = Field(default=True, description="Include phenotypic traits")
    include_measurements: bool = Field(default=True, description="Include measurement data")


class AssayActivityArgs(BaseModel):
    """Arguments for querying assay activities and measurements."""

    entity_id: str = Field(description="Entity ID to query activities for")
    entity_type: Literal["COMPOUND", "GENE", "TARGET"] = Field(description="Type of entity")
    assay_types: list[str] | None = Field(default=None, description="Specific assay types to filter by")
    include_targets: bool = Field(default=True, description="Include target information")
    include_mechanisms: bool = Field(default=True, description="Include mechanism details")


class ComprehensiveGeneArgs(BaseModel):
    """Arguments for comprehensive gene information query."""

    gene_id: str = Field(description="Gene identifier (ENSEMBL:ENSG00000139618 or cbs:ensembl_ENSG00000139618)")
    include_interactions: bool = Field(default=True, description="Include protein-protein interactions")
    include_compounds: bool = Field(default=True, description="Include targeting compounds")
    include_regulatory: bool = Field(default=True, description="Include regulatory relationships")


class ComprehensivePerturbationArgs(BaseModel):
    """Arguments for comprehensive perturbation analysis."""

    perturbation_id: str = Field(description="Perturbation identifier")
    perturbation_type: Literal["COMPOUND", "GENE_KNOCKOUT", "OVEREXPRESSION"] = Field(
        description="Type of perturbation"
    )
    cellular_context: str | None = Field(default=None, description="Filter by cellular context")
    include_selectivity: bool = Field(default=True, description="Include selectivity and off-target data")
    include_resistance: bool = Field(default=True, description="Include resistance mechanisms")
    include_combinations: bool = Field(default=True, description="Include combination effects")
