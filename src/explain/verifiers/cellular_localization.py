from typing import List, Dict
import requests
import os
import json
import google.generativeai as genai

from explain.verifiers.verifier_utils import get_gene_ids

GEMINI_MODEL_NAME = "gemini-1.5-flash"


def CheckSubcellularLocation(entity: str) -> Dict:
    # when translocalisation availble, check whether the predicted translocalisation matches,
    # some translocalisation data are availbale from Uniprot
    # if not, check whether both scls have been reported.
    # Caustion: the celluar context is not taken into account for this version.

    return


def GetSubcellularLocation(entity: str) -> Dict:
    """
    Retrieves the predicted subcellular locations for a given biological entity using multiple data sources.
    Args:
        entity (str): The identifier or name of the biological entity (e.g., protein).
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
    uniprot_scls, trans_scls = scl_from_uniprot(alias_dict["uniProtkbAccession"])
    # get ensembl_id for HPA SCLs
    hpa_scls = scl_from_HPA(alias_dict["HPA"])
    all_scls = set(uniprot_scls + hpa_scls)

    translocations = trans_scls  # +
    # todo: Other methods for translcation events

    return {"locations": all_scls, "translocation_events": translocations}


def scl_from_KG():
    NotImplemented


def get_translocalization():
    NotImplemented


def scl_from_uniprot(uniprot_accession: str) -> List[str]:
    """
    Retrieves the subcellular localization for a given UniProt accession ID.
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
        subcellular_location = data.get("comments", [{}])[0].get(
            "subcellularLocations", []
        )

        if subcellular_location:
            available_scls = [
                loc.get("location", {}).get("value") for loc in subcellular_location
            ]

        # Extract the translocation information
        translocation = data.get("comments", [{}])[0].get("note", [])
        if translocation:
            trans_scls = [loc.get("value") for loc in translocation["texts"]]
        # retrieve trans scls
        for tscl in trans_scls:
            all_trans_scls.append(extract_translocation_details_with_gemini(tscl))

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return available_scls, all_trans_scls


def scl_from_HPA(ensembl_id):
    """
    Retrieves subcellular localization and GO terms for a given
    Ensembl gene ID from the Human Protein Atlas.
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


def extract_translocation_details_with_gemini(text_to_analyze: str) -> list[dict]:
    """
    Uses a Google Gemini LLM to extract translocation events with 'from' and 'to' locations.

    Args:
        text_to_analyze (str): The input text from which to extract information.

    Returns:
        list[dict]: A list of dictionaries, where each dict is
                    {"type": "translocation", "from": "start_location", "to": "end_location"}.
                    Returns an empty list if extraction fails or no data.
    """
    
    genai.configure()
    
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # --- Crafting the Improved Prompt ---
    # We're making the prompt much more specific about the "from" and "to" fields,
    # and providing a direct example matching the desired output format.

    # 
    prompt_instruction = (
        "From the following text, identify all instances where a substance or entity "
        "is described as 'translocating to' or 'translocating into' a specific location. "
        "For each instance, extract the exact phrase indicating translocation "
        "('translocate to' or 'translocating into') and the precise location it moves to. "
        "For each identified event, provide the following details "
        "in a JSON object: "
        "`type` (always 'translocation'), `from` (the origin location), and `to` (the destination location)."
        "If an 'from' location is not explicitly mentioned but implied as an initial state, infer it if logical (e.g., default 'cellular context')."
        "If only a 'to' location is mentioned, the 'from' field can be null or 'unknown'."
        "If a 'from' location is not directly related to a 'to' in a translocation, only provide the 'to' location."
        "Ignore thes event describes 'colocalization in' a location."
        "Return the results as a JSON array of these objects."
        "\n\nExample Output Format:"
        "`[`"
        '`  {"type": "translocation", "from": "cell membrane", "to": "nucleus"},`'
        '`  {"type": "translocation", "from": "unknown", "to": "cytoplasm"},`'
        '`  {"type": "translocation", "from": null, "to": "nucleus"}`'
        "`]`"
        "\n\nOnly return the JSON array. Do not include any other text, explanations, or conversational filler."
    )

    try:
        full_content_for_model = f"{prompt_instruction}\n\nText: {text_to_analyze}"

        response = model.generate_content(
            full_content_for_model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  # Keep very low for precise extraction
                response_mime_type="application/json",  # Ensures JSON output
            ),
        )

        json_output_str = response.text

        # Parse the JSON string into a Python list of dictionaries
        parsed_data = json.loads(json_output_str)

        return parsed_data

    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        if "response" in locals() and response.text:
            print(f"Raw LLM response (for debugging): {response.text}")
        return []
