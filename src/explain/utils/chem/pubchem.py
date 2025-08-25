from typing import Any
from urllib.parse import quote

import requests


class PubChemClient:
    """
    Client for retrieving compound information from PubChem API.
    
    Can resolve molecules by name or SMILES and returns:
      - cid (PubChem Compound ID)
      - name (IUPAC or record title)
      - canonical_smiles
      - inchikey  
      - synonyms (list)
    """

    def __init__(self, base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    def get_compound_info(self,
                         name: str | None = None,
                         smiles: str | None = None) -> dict[str, Any]:
        """
        Resolve a compound from PubChem using either name or SMILES.
        
        Args:
            name: Compound name for search
            smiles: Molecule SMILES for search
            
        Returns:
            Dictionary with compound information or error message
        """
        # Clean inputs
        name = name.strip() if name else None
        smiles = smiles.strip() if smiles else None
        
        # Validate exactly one input
        if (name is None) == (smiles is None):
            return {"error": "Provide exactly one of 'name' or 'smiles'."}
        
        input_kind = "smiles" if smiles is not None else "name"
        query = smiles if input_kind == "smiles" else name

        try:
            # 1) Resolve CID
            cid = self._fetch_cid_by_smiles(query) if input_kind == "smiles" else self._fetch_cid_by_name(query)
            if cid is None:
                return {"input_kind": input_kind, "input": query, "error": "No result found in PubChem."}

            # 2) Fetch properties
            iupac = self._fetch_property_by_cid(cid, "IUPACName")
            title = self._fetch_title_by_cid(cid)
            name_result = iupac or title

            canonical_smiles = self._fetch_property_by_cid(cid, "CanonicalSMILES", smiles=True)
            inchikey = self._fetch_property_by_cid(cid, "InChIKey")
            synonyms = self._fetch_synonyms(cid) or []

            result = {
                "input": query,
                "cid": cid,
                "name": name_result,
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "synonyms": synonyms,
                "input_kind": input_kind,
            }

            if not canonical_smiles:
                result["warning"] = "Record found but SMILES properties were unavailable."

            return result

        except requests.exceptions.RequestException as e:
            return {"input_kind": input_kind, "input": query, "error": f"Unexpected error: {type(e).__name__}: {e}"}

    def get_compound_info_by_inchikey(self, inchikey: str) -> dict[str, Any]:
        """
        Retrieve compound information from PubChem using an InChIKey.
        
        Args:
            inchikey: The InChIKey of the compound
            
        Returns:
            Dictionary with CID and ChEMBL ID, or error message
        """
        url = f"{self.base_url}/inchikey/{inchikey}/xrefs/RegistryID/JSON"

        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                info = data["InformationList"]["Information"][0]
                cid = info.get("CID")
                all_ids = info.get("RegistryID", [])
                chembl_ids = [rid for rid in all_ids if rid.startswith("CHEMBL")]
                return {
                    "CID": cid, 
                    "ChEMBL": chembl_ids[0] if len(chembl_ids) == 1 else chembl_ids,
                    "inchikey": inchikey
                }

            elif response.status_code == 404:
                return {"error": f"Result: Identifier {inchikey} not found in PubChem."}
            else:
                return {"error": f"Received status code {response.status_code} from PubChem."}
        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred during the API request: {e}"}

    def get_iupac_name(self, inchikey: str) -> str | dict:
        """
        Fetch the IUPAC name for a given InChIKey from PubChem.
        
        Args:
            inchikey: The InChIKey of the compound
            
        Returns:
            The IUPAC name if found, otherwise an error message
        """
        url = f"{self.base_url}/inchikey/{inchikey}/property/IUPACName/TXT"
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.text.strip()
            else:
                return {
                    "error": f"Error: Failed to retrieve data. Status code: {response.status_code}\nResponse: {response.text}"
                }
        except requests.exceptions.RequestException as e:
            return {"error": f"Error: A network error occurred: {e}"}

    def get_compound_info_by_synonym(self, synonym: str) -> dict[str, Any]:
        """
        Retrieve compound information from PubChem using a synonym.
        
        Args:
            synonym: A common name or synonym for the compound
            
        Returns:
            Dictionary with CID, SMILES, InChIKey, IUPACName, or error message
        """
        # Step 1: Get the CID from the synonym
        cid_url = f"{self.base_url}/name/{synonym}/cids/TXT"
        try:
            cid_response = requests.get(cid_url, timeout=self.timeout)
            if cid_response.status_code != 200:
                return {"error": f"Could not find a compound ID for '{synonym}'. Status: {cid_response.status_code}"}

            # The response text is the CID
            cid = cid_response.text.strip()
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error during CID retrieval: {e}"}

        # Step 2: Get compound properties using the CID
        props_url = f"{self.base_url}/cid/{cid}/property/SMILES,InChIKey,IUPACName/JSON"
        try:
            props_response = requests.get(props_url, timeout=self.timeout)
            if props_response.status_code != 200:
                return {"error": f"Failed to get properties for CID {cid}. Status: {props_response.status_code}"}
            props_data = props_response.json()

            # Extract the relevant data from the JSON structure
            compound_properties = props_data["PropertyTable"]["Properties"][0]
            return {
                "CID": cid,
                "SMILES": compound_properties.get("SMILES"),
                "InChIKey": compound_properties.get("InChIKey"),
                "IUPACName": compound_properties.get("IUPACName"),
            }
        except (requests.exceptions.RequestException, KeyError) as e:
            return {"error": f"Error retrieving or parsing compound data: {e}"}

    def _fetch_cid_by_name(self, name: str) -> int | None:
        j = self._get_json(f"{self.base_url}/name/{quote(name)}/cids/JSON")
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

    def _fetch_property_by_cid(self, cid: int, prop: str, smiles: bool = False) -> str | None:
        """
        Fetch a property by CID.
        If smiles is True, return any smiles we could find.
        """
        j = self._get_json(f"{self.base_url}/cid/{cid}/property/{prop}/JSON")
        try:
            props = j["PropertyTable"]["Properties"][0]
            if smiles:
                success_props = [v for k, v in props.items() if "smiles" in k.lower() and "smiles" in prop.lower()]
                return success_props[0] if success_props else None
            else:
                return props.get(prop)
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

    def _get_json(self, url: str) -> dict:
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
