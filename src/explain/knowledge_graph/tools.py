"""Tool verifier implementations for knowledge graph queries."""

import asyncio
from typing import Any

from explain.eval.tools._base import ToolVerifier
from explain.eval.tools.knowledge_graph.client import GraphClient
from explain.eval.tools.knowledge_graph.schemas import (
    AssayActivityArgs,
    CellularContextArgs,
    ComprehensiveGeneArgs,
    ComprehensivePerturbationArgs,
    EntityFactsArgs,
    EntityRelationshipArgs,
    PerturbationQueryArgs,
    PhenotypeAssociationArgs,
    PublicationQueryArgs,
    SparqlQueryArgs,
)
from explain.eval.tools.knowledge_graph.service import KnowledgeGraphService


class EntityFactsVerifier(ToolVerifier):
    """Tool for retrieving entity facts."""

    name = "get_entity_facts"
    description = "Get facts about a specific biomedical entity (gene, disease, or chemical)"
    args_schema = EntityFactsArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: EntityFactsArgs) -> tuple[float, dict[str, Any]]:
        try:
            if args.entity_type == "GENE":
                result = asyncio.run(self._kg_service.get_gene_facts(args.entity_id, args.entity_label))
            else:
                return 0.5, {"warning": f"Entity type {args.entity_type} not yet supported for fact retrieval"}

            if result is None:
                return 0.3, {
                    "entity_id": args.entity_id,
                    "entity_type": args.entity_type,
                    "facts": [],
                    "message": "No facts found for this entity",
                }

            label, facts = result
            formatted_facts = [{"relationship": fact.relationship, "value": fact.value} for fact in facts]

            return 1.0, {
                "entity_id": args.entity_id,
                "entity_type": args.entity_type,
                "entity_label": label,
                "facts": formatted_facts,
                "fact_count": len(formatted_facts),
            }

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Entity facts query failed: {str(e)}"}


class EntityRelationshipVerifier(ToolVerifier):
    """Tool for finding relationships between entities."""

    name = "find_entity_relationships"
    description = "Find relationships between biomedical entities (genes, diseases, chemicals)"
    args_schema = EntityRelationshipArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: EntityRelationshipArgs) -> tuple[float, dict[str, Any]]:
        try:
            relationships = asyncio.run(
                self._kg_service.find_relationships(
                    args.source_entity_id, args.source_entity_type, args.target_entity_type
                )
            )

            if not relationships:
                return 0.3, {
                    "source_entity": args.source_entity_id,
                    "source_entity_type": args.source_entity_type,
                    "target_entity_type": args.target_entity_type,
                    "relationships": [],
                    "message": "No relationships found",
                }

            # Format results
            formatted_relationships = []
            for rel in relationships:
                formatted_relationships.append(
                    {
                        "target_entity": rel.target_entity,
                        "target_name": rel.target_name,
                        "relationship_type": rel.relationship_type,
                        "additional_info": rel.additional_info,
                    }
                )

            return 1.0, {
                "source_entity": args.source_entity_id,
                "source_entity_type": args.source_entity_type,
                "target_entity_type": args.target_entity_type,
                "relationships": formatted_relationships,
                "relationship_count": len(formatted_relationships),
            }

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Entity relationship query failed: {str(e)}"}


class PublicationInfoVerifier(ToolVerifier):
    """Tool for querying publication information."""

    name = "get_publication_info"
    description = "Get information about publications including citations and journal details"
    args_schema = PublicationQueryArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: PublicationQueryArgs) -> tuple[float, dict[str, Any]]:
        try:
            articles = asyncio.run(self._kg_service.get_article_information(args.pubmed_ids))

            if not articles:
                return 0.3, {"pubmed_ids": args.pubmed_ids, "articles": {}, "message": "No article information found"}

            # Format the results
            formatted_articles = {}
            for pmid, article in articles.items():
                formatted_articles[pmid] = {
                    "citations": article.citations if args.include_citations else None,
                    "journal": article.journal if args.include_journal else None,
                    "published": article.published.isoformat() if article.published else None,
                }

            return 1.0, {
                "pubmed_ids": args.pubmed_ids,
                "articles": formatted_articles,
                "article_count": len(formatted_articles),
            }

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Publication query failed: {str(e)}"}


class SparqlQueryVerifier(ToolVerifier):
    """Tool for executing custom SPARQL queries."""

    name = "execute_sparql_query"
    description = "Execute a custom SPARQL query against the knowledge graph"
    args_schema = SparqlQueryArgs

    def __init__(self):
        super().__init__()
        self._graph_client = GraphClient()

    def _tool_logic(self, args: SparqlQueryArgs) -> tuple[float, dict[str, Any]]:
        try:
            results = asyncio.run(self._graph_client.query(args.query))

            # Limit results if specified
            if args.limit and len(results) > args.limit:
                results = results[: args.limit]
                truncated = True
            else:
                truncated = False

            return 1.0, {"query": args.query, "results": results, "result_count": len(results), "truncated": truncated}

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"SPARQL query failed: {str(e)}"}


class ComprehensiveGeneVerifier(ToolVerifier):
    """Tool for comprehensive gene information queries."""

    name = "get_comprehensive_gene_info"
    description = "Get ALL available information about a gene in a single comprehensive query"
    args_schema = ComprehensiveGeneArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: ComprehensiveGeneArgs) -> tuple[float, dict[str, Any]]:
        try:
            results = asyncio.run(self._kg_service.get_comprehensive_gene_information(args.gene_id))

            if not results.get("basic_info"):
                return 0.3, {"gene_id": args.gene_id, "message": "No gene information found"}

            # Calculate reward based on data richness
            total_items = (
                len(results.get("synonyms", []))
                + len(results.get("gene_groups", []))
                + len(results.get("expression_profile", []))
                + len(results.get("functional_annotations", {}).get("biological_processes", []))
                + len(results.get("functional_annotations", {}).get("pathways", []))
                + len(results.get("interactions", []))
                + len(results.get("targeting_compounds", []))
                + len(results.get("disease_associations", []))
            )

            if total_items > 50:
                reward = 1.0
            elif total_items > 25:
                reward = 0.9
            elif total_items > 10:
                reward = 0.8
            elif total_items > 0:
                reward = 0.6
            else:
                reward = 0.3

            return reward, results

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Comprehensive gene query failed: {str(e)}"}


class ComprehensivePerturbationVerifier(ToolVerifier):
    """Tool for comprehensive perturbation analysis."""

    name = "get_comprehensive_perturbation_effects"
    description = "Get ALL perturbation effects and associated information in a single comprehensive query"
    args_schema = ComprehensivePerturbationArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: ComprehensivePerturbationArgs) -> tuple[float, dict[str, Any]]:
        try:
            results = asyncio.run(
                self._kg_service.get_comprehensive_perturbation_effects(
                    args.perturbation_id, args.perturbation_type, args.cellular_context
                )
            )

            if not results.get("basic_info"):
                return 0.3, {"perturbation_id": args.perturbation_id, "message": "No perturbation information found"}

            # Calculate reward based on data comprehensiveness
            total_items = (
                len(results.get("direct_targets", []))
                + len(results.get("activity_profiles", []))
                + len(results.get("cellular_contexts", []))
                + len(results.get("phenotypic_effects", []))
                + len(results.get("pathway_effects", []))
                + len(results.get("mechanisms", []))
                + len(results.get("selectivity_profile", []))
                + len(results.get("resistance_mechanisms", []))
            )

            if total_items > 30:
                reward = 1.0
            elif total_items > 15:
                reward = 0.9
            elif total_items > 8:
                reward = 0.8
            elif total_items > 3:
                reward = 0.6
            elif total_items > 0:
                reward = 0.4
            else:
                reward = 0.2

            return reward, results

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Comprehensive perturbation query failed: {str(e)}"}


# Legacy tool verifiers for backward compatibility
class PerturbationQueryVerifier(ToolVerifier):
    """Legacy tool - use ComprehensivePerturbationVerifier for better results."""

    name = "query_perturbation_effects"
    description = "Get perturbation effects (legacy - use get_comprehensive_perturbation_effects)"
    args_schema = PerturbationQueryArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: PerturbationQueryArgs) -> tuple[float, dict[str, Any]]:
        # Delegate to comprehensive method using internal service directly
        comprehensive_args = ComprehensivePerturbationArgs(
            perturbation_id=args.perturbation_id, perturbation_type=args.perturbation_type
        )
        # Use service directly instead of tool verifier to avoid circular dependency
        results = asyncio.run(
            self._kg_service.get_comprehensive_perturbation_effects(
                comprehensive_args.perturbation_id,
                comprehensive_args.perturbation_type,
                comprehensive_args.cellular_context,
            )
        )

        if not results.get("basic_info"):
            return 0.3, {"perturbation_id": args.perturbation_id, "message": "No perturbation information found"}

        return 0.8, results


class CellularContextVerifier(ToolVerifier):
    """Tool for querying cellular context (simplified)."""

    name = "query_cellular_context"
    description = "Get cellular context information"
    args_schema = CellularContextArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: CellularContextArgs) -> tuple[float, dict[str, Any]]:
        try:
            # Use comprehensive gene info which includes expression and context
            results = asyncio.run(self._kg_service.get_comprehensive_gene_information(args.entity_id))

            if not results:
                return 0.3, {"entity_id": args.entity_id, "message": "No cellular context found"}

            # Extract cellular context from comprehensive results
            cellular_data = {
                "entity_id": args.entity_id,
                "expression": results.get("expression_profile", []),
                "pathways": results.get("functional_annotations", {}).get("pathways", []),
                "biological_processes": results.get("functional_annotations", {}).get("biological_processes", []),
                "cellular_components": results.get("functional_annotations", {}).get("cellular_components", []),
            }

            return 0.8, cellular_data

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Cellular context query failed: {str(e)}"}


class PhenotypeAssociationVerifier(ToolVerifier):
    """Tool for querying phenotype associations (simplified)."""

    name = "query_phenotype_associations"
    description = "Get phenotype and disease associations"
    args_schema = PhenotypeAssociationArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: PhenotypeAssociationArgs) -> tuple[float, dict[str, Any]]:
        try:
            # Use comprehensive gene info which includes disease associations
            results = asyncio.run(self._kg_service.get_comprehensive_gene_information(args.entity_id))

            if not results:
                return 0.3, {"entity_id": args.entity_id, "message": "No phenotype associations found"}

            # Extract phenotype data from comprehensive results
            phenotype_data = {
                "entity_id": args.entity_id,
                "entity_type": args.entity_type,
                "diseases": results.get("disease_associations", []),
                "phenotypes": results.get("phenotype_associations", []),
            }

            return 0.8, phenotype_data

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Phenotype association query failed: {str(e)}"}


class AssayActivityVerifier(ToolVerifier):
    """Tool for querying assay activities (simplified)."""

    name = "query_assay_activities"
    description = "Get assay activities and measurements"
    args_schema = AssayActivityArgs

    def __init__(self):
        super().__init__()
        self._kg_service = KnowledgeGraphService()

    def _tool_logic(self, args: AssayActivityArgs) -> tuple[float, dict[str, Any]]:
        try:
            # Use comprehensive methods based on entity type
            if args.entity_type == "COMPOUND":
                results = asyncio.run(
                    self._kg_service.get_comprehensive_perturbation_effects(args.entity_id, "COMPOUND")
                )
                activities = results.get("activity_profiles", [])
            else:
                results = asyncio.run(self._kg_service.get_comprehensive_gene_information(args.entity_id))
                activities = results.get("targeting_compounds", [])

            return 0.8, {"entity_id": args.entity_id, "entity_type": args.entity_type, "activities": activities}

        except (RuntimeError, ValueError, OSError) as e:
            return 0.0, {"error": f"Assay activity query failed: {str(e)}"}


# Remove the AgenticQueryVerifier entirely as requested
