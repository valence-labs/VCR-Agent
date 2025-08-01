import re
from typing import Dict, Optional, List

_EXPECTED_SECTIONS = ["think", "answer", "explain", "dag"]

def check_answer_format(response: str) -> Dict[str, Optional[str]]:
    """
    Checks if the response string contains all the required sections.
    
    Args:
        response: The full response string from the LLM.
        
    Returns:
        A dictionary containing the content of each section, or None if a section is missing.
    """
    parsed_sections = {}
    
    for section in _EXPECTED_SECTIONS:
        match = re.search(f"<{section}>(.*?)<\/{section}>", response, re.DOTALL)
        if match:
            parsed_sections[section] = match.group(1).strip()
        else:
            parsed_sections[section] = None
            
    return parsed_sections

def is_format_correct(parsed_sections: Dict[str, Optional[str]]) -> bool:
    """
    Checks if all required sections are present.
    """
    return all(content is not None for content in parsed_sections.values()) 


def guess_max_turns(claim: str, allowed_primitives: List[str], default_max_turns: int = 5) -> int:
    """
    Guesses the maximum number of turns based on the claim's predicates (primitives).
    It parses the <explain> section of the claim and counts the occurrences
    of the allowed primitives.
    Args:
        claim: The claim to guess the max turns for.
        allowed_primitives: The list of allowed primitives.
        default_max_turns: The default maximum number of turns if the claim does not contain any primitives.

    Returns:
        The maximum number of turns based on the claim's predicates.
    """
    parsed_content = check_answer_format(claim)
    explain_section = parsed_content.get("explain")

    if not explain_section or allowed_primitives is None or len(allowed_primitives) == 0:
        return default_max_turns

    primitives_set = set(allowed_primitives)
    turn_count = 0
    
    for line in explain_section.strip().split('\n'):
        match = re.match(r"(\w+)\s*\(", line.strip())
        if match:
            primitive_name = match.group(1)
            if primitive_name in primitives_set:
                turn_count += 1
    return turn_count if turn_count > 0 else default_max_turns
    
    