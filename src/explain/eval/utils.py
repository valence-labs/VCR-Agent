import re

_EXPECTED_SECTIONS = ["think", "answer", "explain", "dag"]


def check_answer_format(response: str) -> dict[str, str | None]:
    """
    Checks if the response string contains all the required sections.

    Args:
        response: The full response string from the LLM.

    Returns:
        A dictionary containing the content of each section, or None if a section is missing.
    """
    parsed_sections = {}

    for section in _EXPECTED_SECTIONS:
        match = re.search(rf"<{section}>(.*?)<\/{section}>", response, re.DOTALL)
        if match:
            parsed_sections[section] = match.group(1).strip()
        else:
            parsed_sections[section] = None

    return parsed_sections


def is_format_correct(parsed_sections: dict[str, str | None]) -> bool:
    """
    Checks if all required sections are present.
    """
    return all(content is not None for content in parsed_sections.values())


