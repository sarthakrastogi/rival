"""Parsing utility functions."""

import re
import json
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ValidationError


def parse_json_from_llm_response(llm_response: str) -> Union[Dict, List]:
    """
    Parse JSON from an LLM-generated response, removing code block markers and comments.

    Args:
        llm_response: Raw response string from LLM

    Returns:
        Parsed JSON object (dict or list)

    Raises:
        ValidationError: If JSON parsing fails
    """
    try:
        # Remove code block markers
        cleaned_response = re.sub(r"```json|```", "", llm_response)

        # Remove comments (both // and /* */ style)
        cleaned_response = re.sub(
            r"//.*?$|/\*.*?\*/", "", cleaned_response, flags=re.MULTILINE | re.DOTALL
        )

        # Remove extra whitespace
        cleaned_response = cleaned_response.strip()

        return json.loads(cleaned_response)

    except json.JSONDecodeError as e:
        raise ValidationError(f"Failed to parse JSON from LLM response: {str(e)}")


def parse_list_from_llm_response(llm_response: str) -> List[Any]:
    """
    Parse a list specifically from LLM response.

    Args:
        llm_response: Raw response string from LLM

    Returns:
        Parsed list

    Raises:
        ValidationError: If response is not a valid list
    """
    parsed = parse_json_from_llm_response(llm_response)

    if not isinstance(parsed, list):
        raise ValidationError(f"Expected list, got {type(parsed).__name__}")

    return parsed


def parse_dict_from_llm_response(llm_response: str) -> Dict[str, Any]:
    """
    Parse a dictionary specifically from LLM response.

    Args:
        llm_response: Raw response string from LLM

    Returns:
        Parsed dictionary

    Raises:
        ValidationError: If response is not a valid dictionary
    """
    parsed = parse_json_from_llm_response(llm_response)

    if not isinstance(parsed, dict):
        raise ValidationError(f"Expected dict, got {type(parsed).__name__}")

    return parsed


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown-formatted text.

    Args:
        text: Text containing code blocks
        language: Specific language to extract (optional)

    Returns:
        List of code block contents
    """
    if language:
        pattern = f"```{language}\\s*\\n(.*?)\\n```"
    else:
        pattern = r"```(?:\w+)?\s*\n(.*?)\n```"

    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def validate_json_structure(data: Any, expected_structure: Dict[str, type]) -> bool:
    """
    Validate that a parsed JSON object has the expected structure.

    Args:
        data: Parsed JSON data
        expected_structure: Dictionary mapping field names to expected types

    Returns:
        True if structure matches, False otherwise
    """
    if not isinstance(data, dict):
        return False

    for field_name, expected_type in expected_structure.items():
        if field_name not in data:
            return False

        if not isinstance(data[field_name], expected_type):
            return False

    return True
