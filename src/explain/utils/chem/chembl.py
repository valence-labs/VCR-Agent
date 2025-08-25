from typing import Any
from urllib.parse import quote

import requests
from loguru import logger

from explain.utils.chem.cache import DuckDBCache
from explain.utils.chem.mol_utils import to_inchikey


class ChEMBLClient:
    """
    Client for retrieving compound information from ChEMBL API.
    
    Can resolve molecules by name, SMILES, or InChI key and returns:
      - chembl_id
      - name (preferred name)
      - SMILES (canonical)
      - InChIKey
      - synonyms
      - additional metadata
    """

    def __init__(self, base_url: str = "https://www.ebi.ac.uk/chembl/api/data", timeout: float = 20.0, cache_path: str | None = None):
        self.base_url = base_url
        self.timeout = timeout
        self.cache = DuckDBCache(cache_path) if cache_path else None

    def get_compound_info(self,
                         name: str | None = None,
                         smiles: str | None = None,
                         inchikey: str | None = None) -> dict[str, Any]:
        """
        Resolve a compound from ChEMBL using InChIKey (preferred) or name (synonym lookup).
        
        Args:
            name: Molecule name for search
            smiles: Molecule SMILES (will be converted to InChIKey)
            inchikey: Molecule InChI Key (most reliable)
            
        Returns:
            Dictionary with compound information or error message
        """
        # Clean and validate inputs
        name = name.strip() if name else None
        smiles = smiles.strip() if smiles else None
        inchikey = inchikey.strip() if inchikey else None
        
        if not any([name, smiles, inchikey]):
            return {"error": "Provide at least one of 'name', 'smiles', or 'inchikey'."}
        
        # Convert SMILES to InChIKey if provided
        if smiles and not inchikey:
            try:
                inchikey = to_inchikey(smiles)
            except Exception as e:
                logger.error(e)
                return {"error": f"Invalid SMILES string provided: {smiles}"}
        
        # Check cache first
        if self.cache and inchikey:
            cached = self.cache.query(inchikey)
            if cached:
                logger.info(f"Cached compound: {inchikey}")
                payload = cached.get("payload") if isinstance(cached, dict) else None
                return payload if isinstance(payload, dict) else cached
        
        # If no inchikey match, try cache by name via synonyms
        if self.cache and name and not inchikey:
            cached_by_name = self.cache.query_by_synonym(name)
            if cached_by_name:
                payload = cached_by_name.get("payload") if isinstance(cached_by_name, dict) else None
                return payload if isinstance(payload, dict) else cached_by_name
        
        try:
            compound_data: dict | None = None
            
            if inchikey:
                url = f"{self.base_url}/molecule.json?molecule_structures__standard_inchi_key={quote(inchikey)}"
                data = self._get_json(url)
                results = data.get("molecules") or []
                if results:
                    compound_data = results[0]
                else:
                    return {"error": "No result found in ChEMBL."}

            elif name:
                search_url = f"{self.base_url}/molecule/search.json?q={quote(name)}"
                data = self._get_json(search_url)
                results = data.get("molecules") or []
                if not results:
                    return {"error": "No result found in ChEMBL."}
                compound_data = next((r for r in results if r.get("molecule_structures")), results[0])

            else:
                return {"error": "No input provided."}

            # Extract compound information
            compound_info = self._extract_compound_info(compound_data)
            if not compound_info.get("chembl_id"):
                compound_info["warning"] = "Record found but could not extract ChEMBL ID."
            
            # Save to cache if enabled
            if self.cache and compound_info.get("inchikey"):
                try:
                    db_record = {
                        "inchikey": compound_info.get("inchikey"),
                        "chembl_id": compound_info.get("chembl_id"),
                        "name": compound_info.get("name"),
                        "smiles": compound_info.get("smiles"),
                        "synonyms": compound_info.get("synonyms"),
                        "payload": compound_info,
                    }
                    self.cache.save_compound(db_record)
                except Exception as e:
                    logger.error(e)
                    pass
            
            return compound_info

        except requests.exceptions.RequestException as e:
            return {"error": f"Unexpected error: {type(e).__name__}: {e}"}

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

    def _get_json(self, url: str) -> dict:
        """Get JSON from a URL."""
        headers = {"Accept": "application/json"}
        r = requests.get(url, timeout=self.timeout, headers=headers)
        r.raise_for_status()
        return r.json()
