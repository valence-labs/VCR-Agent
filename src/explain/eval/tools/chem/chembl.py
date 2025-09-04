from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from explain.utils.chem.chembl import ChEMBLClient
from explain.utils.chem.mol_utils import to_inchikey

from .._base import ToolVerifier


class ChEMBSearchLArgs(BaseModel):
    """Provide at least one of 'name', 'smiles', or 'inchikey'."""

    name: str | None = Field(default=None, description="Molecule name")
    smiles: str | None = Field(default=None, description="Molecule SMILES")
    inchikey: str | None = Field(default=None, description="Molecule InChI Key")

    @field_validator("name", "smiles", "inchikey")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
        return v

    @model_validator(mode="after")
    def _validate_and_prepare_inputs(self):
        if self.name is None and self.smiles is None and self.inchikey is None:
            raise ValueError("Provide at least one of 'name', 'smiles', or 'inchikey'.")
        if self.smiles:
            try:
                self.inchikey = to_inchikey(self.smiles)
            except Exception as e:
                raise ValueError(f"Invalid SMILES string provided: {self.smiles}") from e

        return self


class ChEMBLResolver(ToolVerifier):
    """
    Resolve a molecule by name, SMILES, or InChI key via ChEMBL.

    Returns feedback with:
      - chembl_id
      - name (preferred name)
      - SMILES
      - InChIKey
      - input_kind ('name', 'smiles', or 'inchikey')
    """

    name = "chembl_chemical_resolver"
    description = "Input 'name', 'smiles', or 'inchikey'; returns ChEMBL compound information."
    args_schema = ChEMBSearchLArgs

    def __init__(self, cache_path: str | None = None):
        super().__init__()
        self.client = ChEMBLClient(cache_path=cache_path)

    def _tool_logic(self, args: ChEMBSearchLArgs) -> tuple[float, dict[str, Any]]:
        """
        Resolve a compound from ChEMBL using the ChEMBLClient.
        Returns the compound information with reward score.
        """
        # Use the ChEMBLClient to get compound information
        result = self.client.get_compound_info(name=args.name, smiles=args.smiles, inchikey=args.inchikey)

        # Handle errors
        if "error" in result:
            return 0.0, result

        # Calculate reward based on presence of key fields
        reward = 1.0 if result.get("chembl_id") and result.get("inchikey") else 0.0

        if reward == 0.0 and "warning" not in result:
            result["warning"] = "Record found but missing key identifiers."

        return reward, result
