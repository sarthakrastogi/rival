"""Configuration settings for the Rival SDK."""

import os
from typing import Optional


class Config:
    """Configuration class for SDK settings."""

    def __init__(self):
        self.api_base_url = os.getenv("RIVAL_API_BASE_URL", "http://localhost:8000")
        self.default_model = os.getenv("RIVAL_DEFAULT_MODEL", "gpt-4.1")
        self.default_limit = int(os.getenv("RIVAL_DEFAULT_LIMIT", "100"))

    @property
    def api_endpoints(self) -> dict:
        """Get API endpoints configuration."""
        return {
            "testcases": f"{self.api_base_url}/sdk/v1/agent/testcases",
            "evaluate": f"{self.api_base_url}/sdk/v1/evaluate/testcase",
            "save_agent": f"{self.api_base_url}/sdk/v1/agents/save",
        }


# Global config instance
config = Config()
