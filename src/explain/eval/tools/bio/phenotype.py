from typing import Any, Literal

from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier
from explain.utils.bio.phenotype import PhenotypeClient
from explain.utils.entity import CompoundEntity, GeneEntity


class PhenotypeArgs(BaseModel):
    """Arguments for checking phenotype."""

    source_entity: str = Field(description="The source entity causing phenotype (e.g., compound, protein, pathway)")
    phenotype: list[str] = Field(description="The phenotypes to check for given the perturbation")
    direction: Literal["induce", "alleviate"] = Field(
        description="Direction of phenotype change. Value: 'induce' or 'alleviate'."
    )
    cell_type: str | None = Field(
        default=None, description="Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes)"
    )


class PhenotypeVerifier(ToolVerifier):
    """Tool for checking if a source entity causes a specific phenotype in specific conditions."""

    name = "check_phenotype"
    description = "Check if a source entity causes a specific phenotype under specific conditions"
    args_schema = PhenotypeArgs

    def __init__(self):
        super().__init__()
        self.phenotype_client = PhenotypeClient()

    def _tool_logic(self, args: PhenotypeArgs) -> tuple[float, dict[str, Any]]:
        """
        Main logic for checking phenotype.
        """
        # Determine entity type and get appropriate identifier
        source_entity = self._resolve_entity(args.source_entity)
        if not source_entity:
            return 0.0, {
                "error": f"Could not resolve entity: {args.source_entity}",
                "verification_status": "NOT_VERIFIED",
            }

        entity_type = type(source_entity).__name__
        cell_type = args.cell_type or "HUVEC"

        # Check phenotype associations
        phenotype_matches = []
        if isinstance(source_entity, GeneEntity):
            gene_id = source_entity.entrez_id or source_entity.symbol
            if gene_id:
                for phenotype in args.phenotype:
                    match = self.phenotype_client.has_phenotype(gene_id, phenotype)
                    phenotype_matches.append(match)

        # Check direction-specific evidence
        direction_verified = False
        if args.direction == "induce":
            # Check for phenoprint effects
            direction_verified = self.phenotype_client.has_phenoprint_effect(args.source_entity, cell_type)
        elif args.direction == "alleviate":
            # Check therapeutic potential for compounds
            if isinstance(source_entity, CompoundEntity) and source_entity.rec_id:
                direction_verified = self.phenotype_client.has_therapeutic_potential(source_entity.rec_id, cell_type)

        # Calculate overall accuracy
        phenotype_accuracy = sum(phenotype_matches) / len(phenotype_matches) if phenotype_matches else 0.0

        # Final reward considers both phenotype match and direction
        reward = phenotype_accuracy if direction_verified else phenotype_accuracy * 0.3

        is_verified = reward > 0.5

        feedback = {
            "source_entity": args.source_entity,
            "entity_type": entity_type,
            "phenotype": {"requested": args.phenotype, "matches": phenotype_matches, "accuracy": phenotype_accuracy},
            "direction": {"requested": args.direction, "verified": direction_verified},
            "cell_type": cell_type,
            "verification_status": "VERIFIED" if is_verified else "NOT_VERIFIED",
            "overall_confidence": reward,
        }

        return reward, feedback

    def _resolve_entity(self, entity_name: str) -> GeneEntity | CompoundEntity | None:
        """Resolve entity name to appropriate entity object."""

        # Try as gene first (if uppercase and short, likely gene symbol)
        if entity_name.isupper() and len(entity_name) <= 10:
            try:
                gene_entity = GeneEntity(name=entity_name)
                if gene_entity.entrez_id or gene_entity.symbol:
                    return gene_entity
            except Exception:
                pass

        # Try as compound
        try:
            compound_entity = CompoundEntity(name=entity_name)
            if compound_entity.inchikey or compound_entity.rec_id:
                return compound_entity
        except Exception:
            pass

        # Default fallback to gene
        try:
            return GeneEntity(name=entity_name)
        except Exception:
            return None
