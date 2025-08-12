import re

import pandas as pd
from datasets import Dataset

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


def guess_max_turns(claim: str, allowed_primitives: list[str], default_max_turns: int = 5) -> int:
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

    for line in explain_section.strip().split("\n"):
        match = re.match(r"(\w+)\s*\(", line.strip())
        if match:
            primitive_name = match.group(1)
            if primitive_name in primitives_set:
                turn_count += 1
    return turn_count if turn_count > 0 else default_max_turns


def as_verifier_dataset(
    df: pd.DataFrame, question_column: str = "question", answer_columns: list[str] | None = None
) -> Dataset:
    """
    Converts a DataFrame with specific experiment columns into a Hugging Face Dataset
    formatted for the 'verifiers' library.

    Args:
        df: A pandas DataFrame containing the experiment results.
            Expected columns include 'question', 'answer', and others to be packed into 'info'.
        question_column: The column name for the question.
        answer_columns: The column names for the answer. If None, the column name will be "response".

    Returns:
        A Hugging Face Dataset object ready for use with verifiers rubrics.
    """
    # First, convert the pandas DataFrame to a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df)
    if answer_columns is None:
        answer_columns = ["response"]
    columns_of_interest = [question_column, *answer_columns]

    def format_for_verifier(example: dict) -> dict:
        """
        Takes a single example from the raw dataset and formats it
        with 'question', 'answer', and 'info' fields.
        """

        # All columns that are not 'question' or 'answer' will be moved into the 'info' dict.
        info_keys = [key for key in example.keys() if key not in columns_of_interest]
        info_dict = {key: example[key] for key in info_keys}

        return {
            "question": example.get(question_column, ""),
            "answer": "\n".join([example.get(col, "") for col in answer_columns]),
            "info": info_dict,
            "task": "hooke-explain-verification",
        }

    # Use the map function to apply the transformation to all examples in the dataset.
    # We remove the original columns to keep the final dataset clean.
    original_columns = dataset.column_names
    formatted_dataset = dataset.map(format_for_verifier, remove_columns=original_columns)

    return formatted_dataset
