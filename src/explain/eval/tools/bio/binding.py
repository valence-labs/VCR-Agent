from typing import Any, Literal

from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier
from explain.eval.tools.bio.entity import CompoundEntity, GeneEntity
from explain.eval.tools.bio.utils import retrieve_from_bigquery


class DTIVerificationArgs(BaseModel):
    """Arguments for checking drug-target interactions."""

    drug: str = Field(description="Drug name or identifier")
    target: str = Field(description="Target protein/gene name")
    interaction_type: Literal["inhibitor", "agonist", "antagonist", "activator", "blocker"] | None = Field(
        default=None, description="Type of interaction (inhibitor|agonist|antagonist|activator|blocker)"
    )
    strength: float | None = Field(default=None, description="Predicted strength of the interaction in uM")

    drug_entity: CompoundEntity = None
    target_entity: GeneEntity = None

    def model_post_init(self, __context__=None):
        self.drug_entity = CompoundEntity(name=self.drug)
        self.target_entity = GeneEntity(name=self.target)


class DTIVerifier(ToolVerifier):
    """Tool for checking drug-target interactions including binding affinity prediction"""

    name = "check_drug_target_interaction"
    description = "Check if a drug interacts with a specific target under given conditions"
    args_schema = DTIVerificationArgs
    dti_sources = ["ChEMBL", "MatchMaker", "Boltz2"]  # to be extended
    score_threshold = 0.5

    def _tool_logic(self, args: DTIVerificationArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking drug-target interactions.
        """
        # get binding score
        binding_score = self._get_binding_scores()

        # todo:
        # interaction_type = self._get_interaction_type()

        is_verified = binding_score > self.score_threshold
        reward = 1.0 if is_verified else 0.0

        feedback = {
            "drug": args.drug,
            "target": args.target,
            "interaction_type": args.interaction_type,
            "strength_uM": args.strength,
            "verification_status": "VERIFIED" if is_verified else "NOT_VERIFIED",
        }
        return reward, feedback

    def _get_binding_scores(self):
        # try get MM score
        dti_res = retrieve_MatchMaker_score(
            target_id=self.args.target_entity.UniprotAC, compound_id=self.drug_entity.REC_ID
        )
        # set score threshold for MM
        self.score_threshold = 2

        if dti_res.shape[0] == 0:
            # run botlz2 precomputed or running botlz2 on demand.
            dti_res = retrieve_boltz2_score(
                target_id=self.args.target_entity.UniprotAC, compound_id=self.drug_entity.smiles
            )
            # set score threshold for boltz2
            self.score_threshold = 0.5

        return dti_res

    def _run_boltz2(self) -> Any:
        return

    def _get_interaction_type(self) -> str:
        return


def retrieve_MatchMaker_score(target_id: str, compound_id: str):
    """Retrive MatchMaker score from datalake"""

    sql = f"""
        SELECT z_score as score
        FROM matchmaker_results__public.matchmaker_rxrx
        WHERE mm.protein = '{target_id}'
        AND mm.rec_id = '{compound_id}'
    """

    dti_res = retrieve_from_bigquery(sql)

    return dti_res.loc[0, "score"]


def retrieve_boltz2_score(target_id: str, compound_id: str):
    # todo: stephan
    return
