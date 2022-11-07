from typing import List, Union


def substring_exists_in_list(
    candidate_substrings: Union[List[str], str], main_str: str
) -> bool:
    """Checks if any of 'candidate_substrings' is a substring of 'main_str'"""
    if not isinstance(candidate_substrings, list):
        candidate_substrings = [candidate_substrings]
    for substr in candidate_substrings:
        if substr.lower() in main_str.lower():
            return True
    return False
