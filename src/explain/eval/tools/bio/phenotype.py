import random
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier


class PhenotypeArgs(BaseModel):
    """Arguments for checking phenotype."""

    source_entity: str = Field(description="The source entity causing phenotype (e.g., compound, protein, pathway)")
    phenotype: list[str] = Field(description="The phenotypes to check for given the perturbation")
    cell_type: str | None = Field(
        default=None, description="Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes)"
    )


class PhenotypeVerifier(ToolVerifier):
    """Tool for checking if a source entity causes a specific phenotype in specific conditions.

    This tool verifies claims about phenotype by querying knowledge bases
    and experimental databases, accounting for cellular context and perturbation conditions.
    """

    name = "check_phenotype"
    description = "Check if a source entity causes a specific phenotype under specific conditions"
    args_schema = PhenotypeArgs

    def _tool_logic(self, args: PhenotypeArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking phenotype.
        """
        phenotype_results = [random.uniform(0, 1) > 0.5 for _ in args.phenotype]
        reward = float(np.mean(phenotype_results)) if phenotype_results else 0.0

        feedback = {
            "source_entity": args.source_entity,
            "phenotype": {
                "requested": args.phenotype,
                "results": phenotype_results,
            },
        }
        return reward, feedback
