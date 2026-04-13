from functools import lru_cache
from typing import Any
from urllib.parse import quote

import requests


class PubChemClient:
    """
    Client for retrieving compound information from PubChem API.

    Can resolve molecules by name or SMILES and returns:
      - cid (PubChem Compound ID)
      - name (IUPAC or record title)
      - smiles
      - inchikey
      - chembldb_id
      - synonyms (list)
    """

    def __init__(self, base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    def get_compound_info(
        self, name: str | None = None, smiles: str | None = None, inchikey: str | None = None
    ) -> dict[str, Any]:
        """
        Resolve a compound from PubChem using either name or SMILES.

        Args:
            name: Compound name for search
            smiles: Molecule SMILES for search
            inchikey: Molecule InChIKey for search
        Returns:
            Dictionary with compound information or error message
        """
        # Clean inputs
        name = name.strip() if name else None
        smiles = smiles.strip() if smiles else None

        # Validate exactly one input
        if (name is None) == (smiles is None) == (inchikey is None):
            return {"error": "Provide exactly one of 'name' or 'smiles' or 'inchikey'."}

        input_kind = "inchikey" if inchikey is not None else "smiles" if smiles is not None else "name"
        query = inchikey if input_kind == "inchikey" else smiles if input_kind == "smiles" else name

        try:
            # 1) Resolve CID
            cid = (
                self._fetch_cid_by_inchikey(query)
                if input_kind == "inchikey"
                else self._fetch_cid_by_smiles(query)
                if input_kind == "smiles"
                else self._fetch_cid_by_name(query)
            )
            if cid is None:
                return {"input_kind": input_kind, "input": query, "error": "No result found in PubChem."}

            # 2) Fetch properties
            property_dict = self._fetch_property_by_cid(cid, ["ConnectivitySMILES", "SMILES", "InChIKey", "IUPACName"])
            synonyms = self._fetch_synonyms(cid) or []
            chembl_id = [x for x in synonyms if x.startswith("CHEMBL")]
            result = {
                "cid": cid,
                "name": property_dict["IUPACName"] or self._fetch_title_by_cid(cid),
                "smiles": property_dict["SMILES"],
                "chembl_id": chembl_id[0] if len(chembl_id) > 0 else None,
                "inchikey": property_dict["InChIKey"],
                "synonyms": synonyms,
            }

            if not property_dict.get("SMILES") or not property_dict.get("ConnectivitySMILES"):
                result["warning"] = "Record found but SMILES properties were unavailable."

            return result

        except requests.exceptions.RequestException as e:
            return {"input_kind": input_kind, "input": query, "error": f"Unexpected error: {type(e).__name__}: {e}"}

    def _fetch_cid_by_name(self, name: str) -> int | None:
        j = self._get_json(f"{self.base_url}/name/{quote(name)}/cids/JSON")
        try:
            return j["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_cid_by_inchikey(self, inchikey: str) -> int | None:
        j = self._get_json(f"{self.base_url}/inchikey/{quote(inchikey)}/cids/JSON")
        try:
            return j["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_cid_by_smiles(self, smiles: str) -> int | None:
        j = self._get_json(f"{self.base_url}/smiles/{quote(smiles)}/cids/JSON")
        try:
            return j["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_property_by_cid(self, cid: int, prop: str | list[str], smiles: bool = False) -> str | None:
        """
        Fetch a property by CID.
        If smiles is True, return any smiles we could find.
        """
        if isinstance(prop, list):
            prop = ",".join(prop)
        url = f"{self.base_url}/cid/{cid}/property/{prop}/JSON"
        j = self._get_json(url)
        try:
            props = j["PropertyTable"]["Properties"][0]

            if smiles:
                success_props = [v for k, v in props.items() if "smiles" in k.lower() and "smiles" in prop.lower()]
                return success_props[0] if success_props else None
            else:
                return dict((p, props.get(p)) for p in prop.split(","))
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_title_by_cid(self, cid: int) -> str | None:
        # Record title (often a common name)
        j = self._get_json(f"{self.base_url}/cid/{cid}/description/JSON")
        try:
            return j["InformationList"]["Information"][0]["Title"]
        except (KeyError, IndexError, TypeError):
            return None

    def _fetch_synonyms(self, cid: int) -> list[str] | None:
        j = self._get_json(f"{self.base_url}/cid/{cid}/synonyms/JSON")
        try:
            return j["InformationList"]["Information"][0]["Synonym"]
        except (KeyError, IndexError, TypeError):
            return None

    @lru_cache(maxsize=1000)  # noqa
    def _get_json(self, url: str) -> dict:
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
