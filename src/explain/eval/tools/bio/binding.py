import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier
from explain.utils.bio.bioref import BiorefClient
from explain.utils.bio.boltz2 import Boltz2Client
from explain.utils.bio.matchmaker import MatchMakerClient
from explain.utils.entity import CompoundEntity, GeneEntity


class DTIVerificationArgs(BaseModel):
    """Arguments for checking drug-target interactions."""

    drug: str = Field(description="Drug name or identifier (e.g inchikey, rec_id, name, etc.)")
    target: str = Field(description="Target protein/gene name as ensembl id")
    interaction_type: Literal["inhibitor", "agonist", "antagonist", "activator", "blocker", "binder"] | None = Field(
        default=None, description="Type of interaction (inhibitor|agonist|antagonist|activator|blocker)"
    )
    strength: float | None = Field(default=None, description="Predicted strength of the interaction in uM")


class DTIVerifier(ToolVerifier):
    """Tool for checking drug-target interactions using bioref database"""

    name = "check_drug_target_interaction"
    description = "Check if a drug interacts with a specific target"
    args_schema = DTIVerificationArgs

    def __init__(self):
        super().__init__()
        self.bioref_client = BiorefClient()
        self.matchmaker_client = MatchMakerClient()
        self.boltz_client = Boltz2Client()

    def _tool_logic(self, args: DTIVerificationArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking drug-target interactions.
        """
        # Create entity objects to get proper identifiers
        failed_to_resolve = False
        try:
            drug_entity = asyncio.run(CompoundEntity.from_id(args.drug))
            target_entity = asyncio.run(GeneEntity.from_id(args.target))
            if target_entity.uniprot_id is None:
                target_entity = asyncio.run(GeneEntity.from_name(args.target))
            if drug_entity.inchikey is None:
                drug_entity = asyncio.run(CompoundEntity.from_name(args.drug))
        except Exception:
            failed_to_resolve = True

        if failed_to_resolve or drug_entity.inchikey is None or target_entity.uniprot_id is None:
            return 0.0, {
                "error": "Failed to resolve drug or target identifiers",
                "drug": args.drug,
                "target": args.target,
            }
        # Get compound identifier (prefer InChI key, fallback to REC_ID)
        compound_id = drug_entity.inchikey or drug_entity.rec_id or args.drug
        # Get target identifier (prefer protein accession, fallback to gene symbol)
        target_id = target_entity.uniprot_id or target_entity.symbol or args.target
        # Get binding information using bioref client
        dti_result = self.bioref_client.get_pli(target_id, compound_id)
        if dti_result.score is None:
            dti_result = self.matchmaker_client.get_pli(target_id, compound_id)
        if dti_result.score is None:
            dti_result = self.boltz_client.get_pli(target_id, compound_id)
        # Check if expected interaction type matches found binding type

        feedback = dti_result.to_dict()

        # Calculate reward based on binding evidence and type match
        base_reward = 1.0 if dti_result.is_binding else 0.0
        type_bonus = 0.2 if dti_result.binding_type == args.interaction_type else 0.0

        feedback.update(
            {
                "compound": args.drug,
                "target": args.target,
                "verification_status": "VERIFIED" if dti_result.is_binding else "NOT_VERIFIED",
            }
        )

        return base_reward + type_bonus, feedback
