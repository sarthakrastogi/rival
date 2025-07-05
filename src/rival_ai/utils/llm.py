"""LLM-related utility functions."""

import re
import json
from typing import List, Dict, Optional

# Optional import with graceful fallback
try:
    from litellm import completion

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    completion = None

from ..config import config
from ..exceptions import APIError


# Default model - this should be defined in your config
DEFAULT_LLM = getattr(config, "default_llm_model", "gpt-3.5-turbo")


def llm_call(messages: List[Dict[str, str]], model: str = DEFAULT_LLM) -> str:
    """
    Calls a language model to generate a response.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model identifier to use for the call

    Returns:
        Generated response content as string

    Raises:
        APIError: If LLM call fails or dependencies are missing
    """
    if not HAS_LITELLM:
        raise APIError(
            "litellm is not installed. Please install it to use LLM functionality."
        )

    try:
        response = completion(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        raise APIError(f"LLM call failed: {str(e)}")


def batch_llm_call(
    message_batches: List[List[Dict[str, str]]],
    model: str = DEFAULT_LLM,
    max_retries: int = 3,
) -> List[str]:
    """
    Make multiple LLM calls in batch.

    Args:
        message_batches: List of message lists for batch processing
        model: Model identifier to use
        max_retries: Maximum number of retries per call

    Returns:
        List of response strings

    Raises:
        APIError: If batch processing fails
    """
    results = []

    for i, messages in enumerate(message_batches):
        for attempt in range(max_retries):
            try:
                response = llm_call(messages, model)
                results.append(response)
                break
            except APIError as e:
                if attempt == max_retries - 1:
                    raise APIError(f"Batch call failed at index {i}: {str(e)}")
                continue

    return results
