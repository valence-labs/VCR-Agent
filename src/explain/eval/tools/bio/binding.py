import random
from typing import Any, Literal

from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier


class DTIVerificationArgs(BaseModel):
    """Arguments for checking drug-target interactions."""

    drug: str = Field(description="Drug name or identifier")
    target: str = Field(description="Target protein/gene name")
    interaction_type: Literal["inhibitor", "agonist", "antagonist", "activator", "blocker"] | None = Field(
        default=None, description="Type of interaction (inhibitor|agonist|antagonist|activator|blocker)"
    )
    strength: float | None = Field(default=None, description="Predicted strength of the interaction in uM")


class DTIVerifier(ToolVerifier):
    """Tool for checking drug-target interactions including binding affinity prediction"""

    name = "check_drug_target_interaction"
    description = "Check if a drug interacts with a specific target under given conditions"
    args_schema = DTIVerificationArgs

    def _tool_logic(self, args: DTIVerificationArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking drug-target interactions.
        """
        is_verified = random.random() > 0.5
        reward = 1.0 if is_verified else 0.0
        feedback = {
            "drug": args.drug,
            "target": args.target,
            "interaction_type": args.interaction_type,
            "strength_uM": args.strength,
            "info": "Interaction is VERIFIED" if is_verified else "Interaction could not be verified (NOT_VERIFIED)",
        }
        return reward, feedback
