from typing import List, Dict
import requests
import os
import json
import google.generativeai as genai

from explain.verifiers.verifier_utils import get_gene_ids

GEMINI_MODEL_NAME = "gemini-1.5-flash"
genai.configure()


def CheckSubcellularLocation(entity: str, from_loc: str, to_loc: str) -> Dict:
    """
    Checks whether a given entity (e.g., protein or gene) is reported to localize in both specified subcellular locations,
    and whether a translocation event between these locations is documented.
    Args:
        entity (str): The name or identifier of the entity (e.g., protein or gene) to check.
        from_loc (str): The source subcellular location.
        to_loc (str): The target subcellular location.
    Returns:
        Dict: A tuple containing two boolean values:
            - scl_bool (bool): True if both from_loc and to_loc are reported subcellular locations for the entity.
            - trans_scl_bool (bool): True if a translocation event between from_loc and to_loc is documented.
    Notes:
        - The function does not account for cellular context.
        - Handles alias resolution for the entity.
        - Future improvements may include plausibility scoring and SCL term standardization.
    """
    # when translocalisation availble, check whether the predicted translocalisation matches,
    # some translocalisation data are availbale from Uniprot
    # if not, check whether both scls have been reported.
    # Caustion: the celluar context is not taken into account for this version.

    # todo: dealing with complex, e.g. SMAD2/3 complex, NF-κB (p65/p50), phosphorylated STAT1,
    #       and ambigous terms e.g. STAT proteins

    # get the alias identifier for the given entity
    entity_alias = get_gene_ids(entity)

    # get SCLs and translocation events
    if entity_alias:
        scl_info = GetSubcellularLocation(entity_alias)

        scl_bool = from_loc in scl_info["locations"] & to_loc in scl_info["locations"]
        trans_scl_bool = False
        if scl_info["translocations"]:
            trans_scl_bool = tanslocation_match(from_loc, to_loc, scl_info["translocations"])

    # todo: maybe add plausibility score
    # todo: SCL term standardization
    return scl_bool, trans_scl_bool


def tanslocation_match(from_loc, to_loc, reference) -> bool:
    """
    Checks if a translocation from a given location to another location exists in the reference data.

    Args:
        from_loc: The source location to check for translocation.
        to_loc: The destination location to check for translocation.
        reference: An iterable of dictionaries, each representing a translocation with 'from' and 'to_loc' keys.

    Returns:
        bool: True if a matching translocation is found in the reference, False otherwise.
    """
    for ref in reference:
        if (reference["from"] == from_loc) and (reference["to_loc"] == to_loc):
            return True
    return False


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
    # other data sources
    NotImplemented


def scl_from_uniprot(uniprot_accession: str) -> List[str]:
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
            all_trans_scls.append(extract_translocation_details_with_gemini(tscl))

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return available_scls, all_trans_scls


def scl_from_HPA(ensembl_id):
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

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

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
        print(parsed_data)
        return parsed_data

    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        if "response" in locals() and response.text:
            print(f"Raw LLM response (for debugging): {response.text}")
        return []
