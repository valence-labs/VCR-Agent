from typing import Any

from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier
from explain.utils.bio.expression import GeneExpressionClient


class GeneExpressionArgs(BaseModel):
    """Arguments for checking gene expression regulation."""

    source_entity: str = Field(description="The source entity causing regulation (e.g., compound, protein, pathway)")
    upregulated_genes: list[str] = Field(default=[], description="List of genes expected to be upregulated")
    downregulated_genes: list[str] = Field(default=[], description="List of genes expected to be downregulated")
    cell_type: str | None = Field(
        default=None, description="Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes)"
    )
    perturbation: str | None = Field(default=None, description="Perturbation condition name")
    reference: str = Field(default="DMSO", description="Reference condition name")


class GeneExpressionVerifier(ToolVerifier):
    """Tool for checking if a source entity regulates gene expression in specific conditions."""

    name = "check_gene_expression"
    description = "Check if genes are regulated as expected under specific perturbation conditions"
    args_schema = GeneExpressionArgs

    def __init__(self):
        super().__init__()
        self.expression_client = GeneExpressionClient()

    def _tool_logic(self, args: GeneExpressionArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking gene regulation.
        """
        if not args.upregulated_genes and not args.downregulated_genes:
            return 0.0, {"error": "At least one gene list (upregulated or downregulated) must be provided"}

        # Use provided perturbation or build from source entity
        perturbation = args.perturbation or args.source_entity
        cell_type = args.cell_type or "HUVEC"

        # Check each expected upregulated gene
        up_results = []
        for gene in args.upregulated_genes:
            is_up = self.expression_client.is_upregulated(gene, perturbation, args.reference, cell_type)
            up_results.append(is_up)

        # Check each expected downregulated gene
        down_results = []
        for gene in args.downregulated_genes:
            is_down = self.expression_client.is_downregulated(gene, perturbation, args.reference, cell_type)
            down_results.append(is_down)

        # Calculate accuracy
        all_results = up_results + down_results
        accuracy = sum(all_results) / len(all_results) if all_results else 0.0

        # Get actual regulated genes for comparison
        actual_up, actual_down = self.expression_client.get_regulated_genes(perturbation, args.reference, cell_type)

        feedback = {
            "source_entity": args.source_entity,
            "perturbation": perturbation,
            "reference": args.reference,
            "cell_type": cell_type,
            "upregulated": {
                "expected": args.upregulated_genes,
                "results": up_results,
                "actual_genes": actual_up[:20],  # Limit for readability
            },
            "downregulated": {
                "expected": args.downregulated_genes,
                "results": down_results,
                "actual_genes": actual_down[:20],  # Limit for readability
            },
            "accuracy": accuracy,
            "verification_status": "VERIFIED" if accuracy > 0.5 else "NOT_VERIFIED",
        }

        return accuracy, feedback
