"""Knowledge Graph module for querying biomedical knowledge.

This module provides comprehensive tools for querying a biomedical knowledge graph
with SPARQL queries to retrieve structured information about genes, diseases,
chemicals, and their relationships.

Main Tools:
-----------
- ComprehensiveGeneVerifier: Get ALL gene information in single query
- ComprehensivePerturbationVerifier: Get ALL perturbation effects in single query
- EntityFactsVerifier: Get basic facts about entities
- EntityRelationshipVerifier: Find relationships between entities
- SparqlQueryVerifier: Execute custom SPARQL queries

Usage:
------
```python
from explain.eval.tools.knowledge_graph import ComprehensiveGeneVerifier

# Get comprehensive gene information
gene_tool = ComprehensiveGeneVerifier()
result = gene_tool.invoke({
    "gene_id": "ENSEMBL:ENSG00000139618",
    "include_interactions": True,
    "include_compounds": True
})
```
"""

from explain.eval.tools.knowledge_graph.client import GraphClient
from explain.eval.tools.knowledge_graph.models import Article, Entity, EntityFact, EntityRelationship
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
from explain.eval.tools.knowledge_graph.tools import (
    AssayActivityVerifier,
    CellularContextVerifier,
    ComprehensiveGeneVerifier,
    ComprehensivePerturbationVerifier,
    EntityFactsVerifier,
    EntityRelationshipVerifier,
    PerturbationQueryVerifier,
    PhenotypeAssociationVerifier,
    PublicationInfoVerifier,
    SparqlQueryVerifier,
)

__all__ = [
    # Core classes
    "GraphClient",
    "KnowledgeGraphService",
    # Data models
    "Entity",
    "Article",
    "EntityFact",
    "EntityRelationship",
    # Argument schemas
    "EntityFactsArgs",
    "EntityRelationshipArgs",
    "PublicationQueryArgs",
    "SparqlQueryArgs",
    "PerturbationQueryArgs",
    "CellularContextArgs",
    "PhenotypeAssociationArgs",
    "AssayActivityArgs",
    "ComprehensiveGeneArgs",
    "ComprehensivePerturbationArgs",
    # Tool verifiers (main interfaces)
    "ComprehensiveGeneVerifier",
    "ComprehensivePerturbationVerifier",
    "EntityFactsVerifier",
    "EntityRelationshipVerifier",
    "PublicationInfoVerifier",
    "SparqlQueryVerifier",
    # Legacy tool verifiers (for backward compatibility)
    "PerturbationQueryVerifier",
    "CellularContextVerifier",
    "PhenotypeAssociationVerifier",
    "AssayActivityVerifier",
]
