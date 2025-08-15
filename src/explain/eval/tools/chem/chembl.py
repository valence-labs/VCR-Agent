from typing import Any
from urllib.parse import quote

import requests
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from .._base import ToolVerifier
from ._cmpd_cache import DuckDBCache
from ._utils import _compute_inchikey


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
                self.inchikey = _compute_inchikey(self.smiles)
            except Exception as e:
                logger.error(e)
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

    timeout: float = 20.0
    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, cache_path: str | None = None):
        super().__init__()
        self.cache = DuckDBCache(cache_path)

    def _tool_logic(self, args: ChEMBSearchLArgs) -> tuple[float, dict[str, Any]]:
        """
        Resolve a compound from ChEMBL using InChIKey (preferred) or name (synonym lookup).
        Returns the curated payload only (the same object that is persisted in the cache's payload column).
        """
        # 1) Cache lookup by inchikey (exact return as stored)
        if getattr(args, "inchikey", None) and getattr(self, "cache", None):
            cached = self.cache.query(args.inchikey)
            if cached:
                logger.info(f"Cached compound: {args.inchikey}")
                payload = cached.get("payload") if isinstance(cached, dict) else None
                return 1.0, payload if isinstance(payload, dict) else cached

        # 1b) If no inchikey match, try cache by name via synonyms
        if getattr(args, "name", None) and getattr(self, "cache", None):
            cached_by_name = self.cache.query_by_synonym(args.name)
            if cached_by_name:
                payload = cached_by_name.get("payload") if isinstance(cached_by_name, dict) else None
                return 1.0, payload if isinstance(payload, dict) else cached_by_name

        compound_data: dict | None = None

        try:
            # 2) Resolve from API
            if args.inchikey:
                ik = args.inchikey
                url = f"{self.base_url}/molecule.json?molecule_structures__standard_inchi_key={quote(ik)}"
                data = self._get_json(url, self.timeout)
                results = data.get("molecules") or []
                if results:
                    compound_data = results[0]
                else:
                    return 0.0, {"error": "No result found in ChEMBL."}

            elif args.name:
                name = args.name
                search_url = f"{self.base_url}/molecule/search.json?q={quote(name)}"
                data = self._get_json(search_url, self.timeout)
                results = data.get("molecules") or []
                if not results:
                    return 0.0, {"error": "No result found in ChEMBL."}
                compound_data = next((r for r in results if r.get("molecule_structures")), results[0])

            else:
                return 0.0, {"error": "No input provided."}

            # 3) Build curated fields (for persistence + downstream); no input/input_kind
            curated = self._extract_compound_info(compound_data)
            reward = 1.0 if curated.get("chembl_id") else 0.0
            if reward == 0.0:
                # Still return what we have, but signal issue
                curated["warning"] = "Record found but could not extract ChEMBL ID."

            if getattr(self, "cache", None) and curated.get("inchikey"):
                try:
                    db_record = {
                        "inchikey": curated.get("inchikey"),
                        "chembl_id": curated.get("chembl_id"),
                        "name": curated.get("name"),
                        "smiles": curated.get("smiles"),
                        "synonyms": curated.get("synonyms"),
                        "payload": curated,
                    }
                    self.cache.save_compound(db_record)
                except Exception as e:
                    logger.error(e)
                    pass
            return reward, curated

        except requests.exceptions.RequestException as e:
            return 0.0, {"error": f"Unexpected error: {type(e).__name__}: {e}"}

    def _extract_compound_info(self, data: dict) -> dict:
        """Extract curated fields from a ChEMBL /molecule payload."""
        ms = data.get("molecule_structures") or {}
        syn_rows = data.get("molecule_synonyms") or []

        def _synonym_candidates(rows):
            out = []
            for row in rows:
                for k in ("synonyms", "molecule_synonym"):
                    v = row.get(k)
                    if isinstance(v, str):
                        s = v.strip()
                        if s:
                            out.append(s)
            return out

        def _uppercase_score(s: str) -> int:
            return sum(1 for c in s if c.isupper())

        candidates = _synonym_candidates(syn_rows)
        best_by_key: dict[str, str] = {}
        for s in candidates:
            key = s.casefold()
            prev = best_by_key.get(key)
            if prev is None:
                best_by_key[key] = s
            else:
                # Prefer more UPPERCASE, then longer, then lexicographically (deterministic)
                new_key = (_uppercase_score(s), len(s), s)
                old_key = (_uppercase_score(prev), len(prev), prev)
                if new_key > old_key:
                    best_by_key[key] = s

        synonyms = sorted(best_by_key.values(), key=str.lower) if best_by_key else None

        is_drug = bool(data.get("therapeutic_flag")) or data.get("max_phase") is not None

        info = {
            # IDs
            "chembl_id": data.get("molecule_chembl_id"),
            "name": data.get("pref_name"),
            "molecule_type": data.get("molecule_type"),
            "is_drug": is_drug,
            "indication_class": data.get("indication_class"),
            "inorganic": bool(data.get("inorganic_flag")),
            "smiles": ms.get("canonical_smiles"),
            "inchikey": ms.get("standard_inchi_key"),
            # Synonyms (deduped w/ casing rule)
            "synonyms": synonyms,
        }

        # Drop empty / None entries
        return {k: v for k, v in info.items() if v not in (None, "", [])}

    def _get_json(self, url: str, timeout_s: float) -> dict:
        """Get JSON from a URL."""
        headers = {"Accept": "application/json"}
        r = requests.get(url, timeout=timeout_s, headers=headers)
        r.raise_for_status()
        return r.json()
