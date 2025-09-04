from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from explain.utils.chem.pubchem import PubChemClient

from .._base import ToolVerifier


class PubChemArgs(BaseModel):
    """Provide exactly one of 'name' or 'smiles'."""

    name: str | None = Field(default=None, description="One of the molecule name")
    smiles: str | None = Field(default=None, description="Molecule SMILES")

    # Strip both inputs if provided
    @field_validator("name", "smiles")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
        return v

    @model_validator(mode="after")
    def _exactly_one(self):
        # Exactly one of name/smiles must be non-None
        if (self.name is None) == (self.smiles is None):
            # (A == B) is True when both are None or both are not None
            raise ValueError("Provide exactly one of 'name' or 'smiles'.")
        return self


class PubChemResolver(ToolVerifier):
    """
    Resolve a molecule by name or SMILES via PubChem.

    Returns feedback with:
      - cid
      - name (IUPAC or record title)
      - SMILES (canonical or isomeric or connectivity)
      - synonyms (list)
      - input_kind ('name' or 'smiles')
    """

    name = "pubchem_chemical_resolver"
    description = "Input either 'name' or 'smiles'; returns PubChem cid, IUPAC name, SMILES, and synonyms."
    args_schema = PubChemArgs

    def __init__(self):
        super().__init__()
        self.client = PubChemClient()

    def _tool_logic(self, args: PubChemArgs) -> tuple[float, dict[str, Any]]:
        """
        Resolve a compound from PubChem using the PubChemClient.
        Returns the compound information with reward score.
        """
        input_kind = "smiles" if args.smiles is not None else "name"
        query = args.smiles if input_kind == "smiles" else args.name

        # Use the PubChemClient to get compound information
        result = self.client.get_compound_info(
            name=args.name if input_kind == "name" else None, smiles=args.smiles if input_kind == "smiles" else None
        )

        # Handle errors
        if "error" in result:
            # Add input information for context
            result.update({"input_kind": input_kind, "input": query})
            return 0.0, result

        # Add input information to successful results
        result.update({"input": query, "input_kind": input_kind})

        # Calculate reward based on presence of SMILES
        reward = 1.0 if result.get("smiles") else 0.0

        if reward == 0.0 and "warning" not in result:
            result["warning"] = "Record found but SMILES properties were unavailable."

        return reward, result
