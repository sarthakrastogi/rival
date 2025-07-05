"""Agent definition models and functionality."""

import requests
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from ..config import Config
from ..exceptions import APIError
from ..utils.agent_representation import generate_agent_representation


class AgentDefinition(BaseModel):
    """Definition of an AI agent for testing purposes."""

    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent's purpose")
    agent_object: Any = Field(..., description="The actual agent object")
    agent_type: str = Field(..., description="Type of agent (e.g., 'langgraph')")
    input_schema: Dict[str, Any] = Field(..., description="Expected input schema")
    output_schema: Dict[str, Any] = Field(..., description="Expected output schema")

    class Config:
        arbitrary_types_allowed = True

    def format_to_string(self) -> str:
        """Format the agent definition as a string representation."""
        try:
            agent_representation = generate_agent_representation(
                agent_object=self.agent_object, agent_type=self.agent_type
            )
        except Exception as e:
            agent_representation = f"Error generating representation: {str(e)}"

        return f"""{self.name}
{self.description}

# Agent Representation
{agent_representation}

# Input Schema:
{self.input_schema}

# Output Schema:
{self.output_schema}"""

    def save_to_project(
        self, api_key: str, project_id: str, overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Save the agent definition to a project via API.

        Args:
            api_key: API key for authentication
            project_id: ID of the project to save to
            overwrite: Whether to overwrite existing agent definition

        Returns:
            API response data

        Raises:
            APIError: If the API request fails
        """
        payload = {
            "project_id": project_id,
            "agent_definition": self.format_to_string(),
            "overwrite": overwrite,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                Config.api_endpoints["save_agent"],
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to save agent definition: {str(e)}")
