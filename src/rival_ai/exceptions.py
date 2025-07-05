"""Custom exceptions for the Rival SDK."""

from typing import Optional, Dict


class RivalSDKError(Exception):
    """Base exception for all Rival SDK errors."""

    pass


class APIError(RivalSDKError):
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


class ValidationError(RivalSDKError):
    """Exception raised for validation errors."""

    pass


class ConfigurationError(RivalSDKError):
    """Exception raised for configuration errors."""

    pass


class TestCaseGenerationError(RivalSDKError):
    """Exception raised for test case generation errors."""

    pass
