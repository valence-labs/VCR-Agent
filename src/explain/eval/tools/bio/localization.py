import asyncio
from typing import Any

from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier
from explain.utils.bio.localization import LocalizationClient
from explain.utils.entity import GeneEntity


class SCLArgs(BaseModel):
    """Arguments for checking subcellular localization and translocation."""

    protein: str = Field(description="Protein/gene name")
    from_loc: str | None = Field(default=None, description="Original location")
    to_loc: str = Field(description="Translocated location")
    mechanism: str | None = Field(
        default=None, description="Effect on the translocation, such as 'induce', 'block', etc."
    )
    modification: str | None = Field(
        default=None,
        description="Presence of any post translation modification such as 'phosphorylation', 'dimerization' etc.",
    )


class SCLVerifier(ToolVerifier):
    """Tool for checking subcellular localization and translocation events."""

    name = "check_subcellular_translocation"
    description = "Check if a protein translocates from one subcellular location to another"
    args_schema = SCLArgs

    def __init__(self):
        super().__init__()
        self.localization_client = LocalizationClient()

    def _tool_logic(self, args: SCLArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking subcellular translocation.
        """
        # Create protein entity to get UniProt ID
        try:
            protein_entity = asyncio.run(GeneEntity.from_name(args.protein))
            if protein_entity.uniprot_id is None:
                protein_entity = asyncio.run(GeneEntity.from_id(args.protein))
            uniprot_id = protein_entity.uniprot_id
        except Exception:
            uniprot_id = None

        if not uniprot_id:
            return 0.0, {
                "error": f"Could not resolve UniProt ID for protein: {args.protein}",
                "protein": args.protein,
                "verification_status": "NOT_VERIFIED",
            }

        # Get all known locations for the protein
        all_locations = self.localization_client.get_locations(uniprot_id)
        can_translocate = self.localization_client.can_translocate(
            uniprot_id, args.from_loc, args.to_loc, known_locations=all_locations
        )

        reward = float(can_translocate)
        # TODO: Add mechanism and modification

        feedback = {
            "protein": args.protein,
            "uniprot_id": uniprot_id,
            "from_loc": args.from_loc,
            "to_loc": args.to_loc,
            "modification": args.modification,
            "all_locations": all_locations,
            "can_translocate": can_translocate,
            "verification_status": "VERIFIED" if reward > 0.5 else "PARTIAL" if reward > 0.0 else "NOT_VERIFIED",
        }

        return reward, feedback
