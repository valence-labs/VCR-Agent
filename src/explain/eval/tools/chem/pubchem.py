from typing import Any
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field, field_validator, model_validator

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

    timeout: float = 10.0

    base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    def _tool_logic(self, args: PubChemArgs) -> tuple[float, dict[str, Any]]:
        input_kind = "smiles" if args.smiles is not None else "name"
        q = args.smiles if input_kind == "smiles" else args.name
        t = self.timeout

        try:
            # 1) Resolve CID
            cid = self._fetch_cid_by_smiles(q, t) if input_kind == "smiles" else self._fetch_cid_by_name(q, t)
            if cid is None:
                return 0.0, {"input_kind": input_kind, "input": q, "error": "No result found in PubChem."}

            # 2) Fetch properties
            iupac = self._fetch_property_by_cid(cid, "IUPACName", t)
            title = self._fetch_title_by_cid(cid, t)
            name = iupac or title

            can_smi = self._fetch_property_by_cid(cid, "CanonicalSMILES", t, smiles=True)

            synonyms = self._fetch_synonyms(cid, t) or []

            payload = {
                "input": q,
                "cid": cid,
                "name": name,
                "canonical_smiles": can_smi,
                "synonyms": synonyms,
                "input_kind": input_kind,
            }

            reward = 1.0 if can_smi else 0.0
            if reward == 0.0:
                payload["warning"] = "Record found but SMILES properties were unavailable."

            return reward, payload

        except requests.exceptions.RequestException as e:  # noqa
            return 0.0, {"input_kind": input_kind, "input": q, "error": f"Unexpected error: {type(e).__name__}: {e}"}

    def _fetch_cid_by_name(self, name: str, timeout_s: float) -> int | None:
        j = self._get_json(f"{self.base_url}/name/{quote(name)}/cids/JSON", timeout_s)
        try:
            return j["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_cid_by_smiles(self, smiles: str, timeout_s: float) -> int | None:
        j = self._get_json(f"{self.base_url}/smiles/{quote(smiles)}/cids/JSON", timeout_s)
        try:
            return j["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_property_by_cid(self, cid: int, prop: str, timeout_s: float, smiles: bool = True) -> str | None:
        """
        Fetch a property by CID.
        If smiles is True, return any smiles we could find.
        """
        j = self._get_json(f"{self.base_url}/cid/{cid}/property/{prop}/JSON", timeout_s)
        try:
            props = j["PropertyTable"]["Properties"][0]
            if smiles:
                success_props = [v for k, v in props.items() if "smiles" in k.lower() and "smiles" in prop.lower()]
                return success_props[0] if success_props else None
            else:
                return props[prop]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_title_by_cid(self, cid: int, timeout_s: float) -> str | None:
        # Record title (often a common name)
        j = self._get_json(f"{self.base_url}/cid/{cid}/description/JSON", timeout_s)
        try:
            return j["InformationList"]["Information"][0]["Title"]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_synonyms(self, cid: int, timeout_s: float) -> list[str] | None:
        j = self._get_json(f"{self.base_url}/cid/{cid}/synonyms/JSON", timeout_s)
        try:
            return j["InformationList"]["Information"][0]["Synonym"]
        except (KeyError, IndexError, TypeError):
            return None

    def _get_json(self, url: str, timeout_s: float) -> dict:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        return r.json()
