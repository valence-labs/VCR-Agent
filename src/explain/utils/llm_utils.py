from explain.llm import create_client
from explain.llm._client import LLMClient


def get_entity_type(input: str, client: LLMClient | None = None) -> str:
    """
    Classify input text as 'compound', 'protein', or 'pathway' using an LLM.

    Args:
        input (str): Input text to classify.
        client (LLMClient, optional): LLM client instance.

    Returns:
        str: The predicted entity type ('compound', 'protein', or 'pathway').
    """
    if not client:
        client = create_client(provider="litellm", model="gemini-2.5-flash")

    llm_instruction = """
        You are an expert bioinformatics entity classifier. Your task is to analyze the input text and determine if it represents a `compound`, a `protein`, or a `pathway`.

        Follow these classification guidelines:
        1.  **compound**: Refers to a chemical substance, small molecule, or drug. Examples include "aspirin", "glucose", "ethanol", "metformin".
        2.  **protein**: Typically a gene name or its protein product. Often represented by a capitalized acronym that may include numbers. Examples include "MTOR1", "TP53", "EGFR", "AKT1".
        3.  **pathway**: Describes a biological process, a series of interactions, or a metabolic route. Often contains keywords like "signaling", "pathway", "cascade", "metabolism", or "cycle". Examples include "MTOR signaling", "Glycolysis", "Apoptosis", "p53 signaling pathway".

        Your output MUST be ONLY the single, lowercase word for the determined type: `compound`, `protein`, or `pathway`. Do not provide any explanation or additional text.

        ---
        **Examples:**

        Input: "aspirin"
        Output: "compound"

        Input: "MTOR1"
        Output: "protein"

        Input: "MTOR signaling"
        Output: "pathway"

        Input: "Krebs cycle"
        Output: "pathway"

        Input: "BRCA2"
        Output: "protein"
        ---

    """

    input_text = f"""

        **Perform the classification for the following input:**

        Input: "{input}"

    """

    content = llm_instruction + input_text

    res = client.generate([{"role": "user", "content": content}])

    return res.content


def compare_phenotypes(phenotype_1: str, phenotype_2: str, client: LLMClient | None = None) -> bool | str:
    """
    Compare two phenotype descriptions and determine if they describe the same cellular phenotype.

    Args:
        phenotype_1 (str): First phenotype description.
        phenotype_2 (str): Second phenotype description.
        client (optional): LLM client instance.

    Returns:
        bool or str: True if phenotypes are equivalent, False if not, or LLM response otherwise.
    """
    if not client:
        client = create_client(provider="litellm", model="gemini-2.5-flash")

    llm_instruction = """
        Task: Decide whether the following two texts describe the same cellular phenotype,
        even if they use different wording or synonyms.
        Consider them the same if their meanings are equivalent.
        Only return true or false without explanation.
    
    Instructions:

        Carefully read Text 1 and Text 2.

        Identify all terms related to cellular phenotypes, such as cell morphology, function, proliferation, death, or signaling.

        Compare the terms identified in both texts.

        Return true if the terms are synonymous or describe a cohesive set of features that represent the same cellular state or process.

        Return false if the terms describe distinct or unrelated cellular phenotypes.

        Examples:

        Example 1: True

        Text 1: "Analysis revealed a significant increase in programmed cell death, leading to a reduction in cell count."

        Text 2: "The biopsy showed widespread apoptosis throughout the tissue."

        Output: true

        Example 2: False

        Text 1: "The cells displayed a highly disorganized cytoskeleton and impaired motility."

        Text 2: "Mitochondrial swelling and reduced ATP production were observed."

        Output: false
    """

    input_text = f"""

    **Perform the comparison for the following inputs:**

    Text 1: "{phenotype_1}"
    
    Text 2: "{phenotype_2}"

    """

    content = llm_instruction + input_text

    res = client.generate([{"role": "user", "content": content}])

    if isinstance(res, str):
        if res.strip().lower() == "false":
            return False
        elif res.strip().lower() == "true":
            return True
        else:
            return res
    elif hasattr(res, "content"):
        val = res.content.strip().lower()
        if val == "false":
            return False
        elif val == "true":
            return True
        else:
            return res.content
    else:
        return res
