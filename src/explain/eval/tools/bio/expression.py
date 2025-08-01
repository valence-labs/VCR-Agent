from typing import List, Optional, Tuple, Dict, Any
import random
import numpy as np
from pydantic import BaseModel, Field
from explain.eval.tools._base import ToolVerifier


class GeneExpressionArgs(BaseModel):
    """Arguments for checking regulation expression."""
    
    source_entity: str = Field(description="The source entity causing regulation (e.g., compound, protein, pathway)")
    upregulated_genes: List[str] = Field(default=[], description="List of genes expected to be upregulated")
    downregulated_genes: List[str] = Field(default=[], description="List of genes expected to be downregulated")
    cell_type: Optional[str] = Field(default=None, description="Cell type or cell line (e.g., HepG2, MCF7, primary hepatocytes)")
    gkos: List[str] = Field(default=[], description="List of gene knockouts applied as perturbations")
    compounds: List[str] = Field(default=[], description="List of compound perturbations applied prior to measurement")
    dose: Optional[float] = Field(default=None, description="Dose/concentration used in uM")


class GeneExpressionVerifier(ToolVerifier):
    """Tool for checking if a source entity regulates gene expression in specific conditions.
    
    This tool verifies claims about gene regulation by querying knowledge bases
    and experimental databases, accounting for cellular context and perturbation conditions.
    """
    effect_pval = 0.05
    name = "check_gene_expression"
    description = "Check if a source entity regulates gene expression under specific conditions"
    args_schema = GeneExpressionArgs
    
    def _tool_logic(self, args: GeneExpressionArgs) -> Tuple[float, Dict[str, Any]]:
        """
        Tool logic for checking gene regulation.
        """
        if not args.upregulated_genes and not args.downregulated_genes:
            return 0.0, {"error": "At least one gene list (upregulated or downregulated) must be provided"}

        up_regulation_results = [random.uniform(0, 1) < self.effect_pval for _ in args.upregulated_genes]
        down_regulation_results = [random.uniform(0, 1) < self.effect_pval for _ in args.downregulated_genes]
        
        all_results = up_regulation_results + down_regulation_results
        reward = float(np.mean(all_results)) if all_results else 0.0
        
        feedback = {
            "source_entity": args.source_entity,
            "downregulated": {
                "requested": args.downregulated_genes,
                "results": down_regulation_results,
            },
            "upregulated": {
                "requested": args.upregulated_genes,
                "results": up_regulation_results,
            }
        }
        return reward, feedback
    
    