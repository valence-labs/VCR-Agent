"""Data models for knowledge graph entities and relationships."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal


@dataclass(frozen=True)
class Entity:
    """Represents a biomedical entity in the knowledge graph."""

    id: str
    type: Literal["DISEASE", "GENE", "CHEMICAL"]


@dataclass(frozen=True)
class Article:
    """Represents publication information."""

    citations: int
    journal: str
    published: date | None


@dataclass(frozen=True)
class EntityFact:
    """Represents a fact about an entity with rich contextual information."""

    relationship: str
    value: str
    description: str = ""
    category: str = ""
    source: str = ""
    confidence: str = ""
    additional_context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityRelationship:
    """Represents a relationship between entities with rich context."""

    target_entity: str
    target_name: str
    relationship_type: str
    target_description: str = ""
    relationship_description: str = ""
    evidence_level: str = ""
    source_database: str = ""
    additional_info: dict[str, Any] = field(default_factory=dict)
