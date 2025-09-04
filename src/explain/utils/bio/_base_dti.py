"""Base interface for drug-target interaction clients."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class DTIResult:
    """Standardized result for drug-target interaction queries."""

    def __init__(
        self,
        target_id: str | None = None,
        compound_id: str | None = None,
        score: float | None = None,
        unit: str | None = None,
        method: str = "unknown",
        binding_type: str = "binding",
        confidence: float = 0.0,
        is_binding: bool = False,
        raw_data: Any = None,
    ):
        self.target_id = target_id
        self.compound_id = compound_id
        self.score = score
        self.unit = unit
        self.method = method
        self.binding_type = binding_type  # "binding", "inhibitor", "agonist", etc.
        self.confidence = confidence
        self.is_binding = is_binding
        self.raw_data = raw_data

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target_id": self.target_id,
            "compound_id": self.compound_id,
            "score": self.score,
            "unit": self.unit,
            "method": self.method,
            "binding_type": self.binding_type,
            "confidence": self.confidence,
            "is_binding": self.is_binding,
        }


class BaseDTIClient(ABC):
    """Abstract base class for drug-target interaction clients."""

    def __init__(self, binding_threshold: Optional[float] = None):
        self.binding_threshold = binding_threshold

    @abstractmethod
    def get_pli(self, target_id: str, compound_id: str, **kwargs) -> DTIResult:
        """Get binding score for target-compound pair."""
        pass

    def is_binding(self, target_id: str, compound_id: str, **kwargs) -> bool:
        """Check if target and compound are predicted to bind."""
        result = self.get_pli(target_id, compound_id, **kwargs)
        return result.is_binding

    @abstractmethod
    def get_targets(self, compound_id: str, limit: int = 10, **kwargs) -> list[str]:
        """Get top predicted targets for a compound."""
        pass

    @abstractmethod
    def get_compounds(self, target_id: str, limit: int = 10, **kwargs) -> list[str]:
        """Get top predicted compounds for a target."""
        pass

    def get_full_pli_info(self, target_id: str, compound_id: str, **kwargs) -> dict[str, Any]:
        """Get full PLI information for a target-compound pair."""
        raise NotImplementedError("get_full_pli_info is not implemented")
