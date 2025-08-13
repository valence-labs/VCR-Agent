import json
from typing import Any

import requests
from loguru import logger
from pydantic import BaseModel, Field

from explain.eval.tools._base import ToolVerifier
from explain.eval.tools.bio.entity import GeneEntity
from explain.llm import create_client


class SCLArgs(BaseModel):
    """Arguments for checking drug-target interactions."""

    protein: str = Field(description="Protein/gene name")
    protein_entity: GeneEntity = None
    from_loc: str | None = Field(default=None, description="Orignial location")
    to_loc: str = Field(description="Translocated location")
    mechanism: str | None = Field(
        default=None, description="Effect on the translocation, such as 'induce', 'block', etc."
    )
    modification: str | None = Field(
        default=None,
        description="Presence of any post translation modification such as 'phosphorylation', 'dimerization' etc.",
    )

    def _get_entity_aliases(self):
        # get protein entity and identifiers
        protein_entity = GeneEntity(name=self.protein)
        self.protein_entity = protein_entity.retrieve_identifiers()


class SCLVerifier(ToolVerifier):
    """Tool for checking drug-target interactions including binding affinity prediction"""

    name = "check_subcellular_translocation"
    description = "Check if a protein translocate from one SCL to another under given conditions"
    args_schema = SCLArgs
    client = create_client(provider="openai", model="gpt-4.1")

    def _tool_logic(self, args: SCLArgs) -> tuple[float, dict[str, Any]]:
        """
        Tool logic for checking drug-target interactions.
        """
        scl_bool, trans_scl_bool = self._get_scl_annotations(args)
        is_verified = trans_scl_bool

        reward = int(scl_bool) * 0.3 + int(trans_scl_bool) * 0.7

        feedback = {
            "protein_entity": args.protein_entity,
            "from_loc": args.from_loc,
            "to_loc": args.to_loc,
            "mechanism": args.mechanism,
            "modification": args.modification,
            "verification_status": "VERIFIED" if is_verified else "NOT_VERIFIED",
        }
        return reward, feedback

    def _get_scl_annotations(self, args):
        scl_bool, trans_scl_bool = False, False

        if args.protein_entity:
            scl_info = self._getSCL(args.protein_entity)

            scl_bool = (args.from_loc in scl_info["locations"]) & (args.to_loc in scl_info["locations"])
            trans_scl_bool = False
            if len(scl_info["translocations"]) > 0:
                trans_scl_bool = self._tanslocation_match(args, scl_info["translocations"])
        else:
            logger.info("Make sure to run the entity retrieval.")
        # todo: maybe add plausibility score
        # todo: SCL term standardization
        return scl_bool, trans_scl_bool

    def _tanslocation_match(self, args, reference) -> bool:
        """
        Checks if a translocation from a given location to another location exists in the reference data.

        Args:
            from_loc: The source location to check for translocation.
            to_loc: The destination location to check for translocation.
            reference: An iterable of dictionaries, each representing a translocation with 'from' and 'to_loc' keys.

        Returns:
            bool: True if a matching translocation is found in the reference, False otherwise.
        """
        # todo: use smarter way for the comparison, e.g. LLM based matching

        from_loc_lower = args.from_loc.lower()
        to_loc_lower = args.to_loc.lower()
        has_modification = args.modification is not None
        is_phosphorylation_mod = has_modification and "phosphorylation" in args.modification

        for ref in reference:
            # Use .get() for safer dictionary access
            ref_from = ref.get("from")
            ref_to_loc = ref.get("to_loc")

            # Skip if required keys are missing or values are empty
            if not ref_from or not ref_to_loc:
                continue

            if ref_from.lower() == from_loc_lower and ref_to_loc.lower() == to_loc_lower:
                # Check for modification only if it's specified in args
                if has_modification:
                    if is_phosphorylation_mod and ref.get("is_phosphorylated"):
                        return True
                else:
                    # No modification specified, so location match is sufficient
                    return True

        return False

    def _getSCL(self, entity_alias: GeneEntity) -> dict:
        """
        Retrieves the predicted subcellular locations for a given biological entity using multiple data sources.
        Args:
            entity_alias (dict): The identifier or name of the biological entity (e.g., protein).
        Returns:
            Dict: A dictionary containing:
                - "locations": A set of predicted subcellular locations aggregated from UniProt and Human Protein Atlas (HPA).
                - "translocation_events": A list of detected translocation events (currently empty).
        Notes:
            - Utilizes UniProt and HPA data sources to infer subcellular localization.
            - Requires `alias_dict` to map entity identifiers to UniProt and HPA accessions.
            - Helper functions `scl_from_uniprot` and `scl_from_HPA` are used to fetch locations from respective sources.
        """
        # use KG, uniprot, human preotein atlas

        # get uniprot ID for uniprot SCLs and translocations
        uniprot_scls, trans_scls = self._scl_from_uniprot(entity_alias.UniprotAC)
        # get ensembl_id for HPA SCLs
        hpa_scls = self._scl_from_HPA(entity_alias.HPA)
        all_scls = set(uniprot_scls + hpa_scls)
        all_scls = [s.lower() for s in all_scls]

        translocations = trans_scls  # +
        # todo: Other methods for translcation events

        return {"locations": all_scls, "translocations": translocations}

    def _scl_from_uniprot(self, uniprot_accession: str) -> list[str]:
        """
        Retrieves subcellular localization (SCL) information for a given UniProt accession ID using the UniProt REST API.

        Args:
            uniprot_accession (str): The UniProt accession ID of the protein.

        Returns:
            Tuple[List[str], List[Any]]:
                - available_scls: A list of subcellular localization names extracted from the UniProt entry.
                - all_trans_scls: A list of translocation details extracted and processed from the UniProt entry.

        Notes:
            - This function queries the UniProt API for subcellular location comments.
            - Translocation details are further processed using the `extract_translocation_details_with_gemini` function.
            - In case of a request error, an error message is printed and empty lists are returned.
        """
        # Todo: return GO terms for SCLs

        # The UniProt API endpoint for retrieving specific fields for a protein
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}?fields=cc_subcellular_location"

        available_scls = []
        all_trans_scls = []

        try:
            # Make the GET request to the API
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            # Parse the JSON response
            data = response.json()

            # Extract the subcellular location information
            subcellular_location = data.get("comments", [{}])[0].get("subcellularLocations", [])

            if subcellular_location:
                available_scls = [loc.get("location", {}).get("value") for loc in subcellular_location]

            # Extract the translocation information
            translocation = data.get("comments", [{}])[0].get("note", [])
            if translocation:
                trans_scls = [loc.get("value") for loc in translocation["texts"]]
            # retrieve trans scls
            for tscl in trans_scls:
                all_trans_scls.extend(self._extract_translocation_details_with_llm(tscl))

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

        return available_scls, all_trans_scls

    def _scl_from_HPA(self, ensembl_id):
        """
        Retrieves subcellular localization information for a given Ensembl gene ID from the Human Protein Atlas (HPA).

        Args:
            ensembl_id (str): The Ensembl gene ID for which to retrieve subcellular localization data.

        Returns:
            list: A list of subcellular localizations associated with the gene, or an empty list if none are found or an error occurs.

        Notes:
            - This function queries the HPA API and expects the response to contain a "Subcellular location" field.
            - If the request fails or the field is missing, an empty list is returned.
        """
        # The HPA API endpoint for retrieving data for a specific gene
        url = f"https://www.proteinatlas.org/{ensembl_id}.json"

        available_scls = []

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            data = response.json()

            available_scls = data.get("Subcellular location", [])

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

        return available_scls

    def _extract_translocation_details_with_llm(self, text_to_analyze: str) -> list[dict]:
        """
        Uses a LLM to extract translocation events with 'from' and 'to' locations.

        Args:
            text_to_analyze (str): The input text from which to extract information.

        Returns:
            list[dict]: A list of dictionaries, where each dict is
                        {"type": "translocation", "from": "start_location", "to": "end_location"}.
                        Returns an empty list if extraction fails or no data.
        """

        # --- Crafting the Improved Prompt ---
        # We're making the prompt much more specific about the "from" and "to" fields,
        # and providing a direct example matching the desired output format.

        prompt_instruction = (
            "From the following text, identify all instances where a substance or entity "
            "is described as 'translocating to' or 'translocating into' a specific location. "
            "For each instance, extract the exact phrase indicating translocation "
            "('translocate to' or 'translocating into') and the precise location it moves to. "
            "For each identified event, provide the following details "
            "in a JSON object: "
            "`from` (the origin location), `to` (the destination location), and `is_phosphorylated` (the phosphorylation boolean)."
            "If a 'from' location is not explicitly mentioned but implied as an initial state, infer it if logical (e.g., default 'cellular context')."
            "If only a 'to' location is mentioned, the 'from' field can be null."
            "If a 'from' location is not directly related to a 'to' in a translocation, only provide the 'to' location."
            "If a 'is_phosphorylated' is not mentioned, the 'is_phosphorylated' field can be null. "
            "Ignore thes event describes 'colocalization in' a location."
            "Return the results as a JSON array of these objects."
            "\n\nExample Output Format:"
            "`[`"
            '`  { "from": "cell membrane", "to": "nucleus", "is_phosphorylated": null},`'
            '`  { "from": null, "to": "cytoplasm", "is_phosphorylated": True},`'
            '`  { "from": null, "to": "nucleus", "is_phosphorylated": null}`'
            "`]`"
            "\n\nOnly return the JSON array. Do not include any other text, explanations, or conversational filler."
        )

        try:
            full_content_for_model = f"{prompt_instruction}\n\nText: {text_to_analyze}"

            response = self.client.generate([{"role": "user", "content": full_content_for_model}])

            json_output_str = response.to_dict()["messages"][-1]["content"]

            # Parse the JSON string into a Python list of dictionaries
            parsed_data = json.loads(json_output_str)
            return parsed_data

        except Exception as e:
            print(f"An error occurred during extraction: {e}")
            if "response" in locals() and response.text:
                print(f"Raw LLM response (for debugging): {response.text}")
            return []


# def scl_from_KG():
#     NotImplemented


# def get_translocalization():
#     # other data sources
#     NotImplemented
