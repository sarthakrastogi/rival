"""Custom exceptions for Rival AI."""

from typing import Optional, Dict


class RivalAIError(Exception):
    """Base exception for all Rival AI errors."""

    pass


class APIError(RivalAIError):
    """Exception raised for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ValidationError(RivalAIError):
    """Exception raised for validation errors."""

    pass


class ConfigurationError(RivalAIError):
    """Exception raised for configuration errors."""

    pass


class TestCaseGenerationError(RivalAIError):
    """Exception raised for test case generation errors."""

    pass
