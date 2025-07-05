"""Tests for agent definition functionality."""

import pytest
from unittest.mock import Mock, patch

from rival_ai.core.agent_definition import AgentDefinition
from rival_ai.exceptions import APIError


class TestAgentDefinition:
    """Test cases for AgentDefinition class."""

    def test_agent_definition_creation(self):
        """Test creating an agent definition."""
        agent_def = AgentDefinition(
            name="Test Agent",
            description="A test agent",
            agent_object=Mock(),
            agent_type="langgraph",
            input_schema={"input": "string"},
            output_schema={"output": "string"},
        )

        assert agent_def.name == "Test Agent"
        assert agent_def.description == "A test agent"
        assert agent_def.agent_type == "langgraph"

    def test_format_to_string(self):
        """Test formatting agent definition to string."""
        agent_def = AgentDefinition(
            name="Test Agent",
            description="A test agent",
            agent_object=Mock(),
            agent_type="langgraph",
            input_schema={"input": "string"},
            output_schema={"output": "string"},
        )

        formatted = agent_def.format_to_string()
        assert "Test Agent" in formatted
        assert "A test agent" in formatted
        assert "Input Schema:" in formatted
        assert "Output Schema:" in formatted

    @patch("rival_ai.core.agent_definition.requests.post")
    def test_save_to_project_success(self, mock_post):
        """Test successful save to project."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True, "message": "Saved"}
        mock_post.return_value = mock_response

        agent_def = AgentDefinition(
            name="Test Agent",
            description="A test agent",
            agent_object=Mock(),
            agent_type="langgraph",
            input_schema={"input": "string"},
            output_schema={"output": "string"},
        )

        result = agent_def.save_to_project(
            "test-api-key", "test-project", overwrite=True
        )

        assert result["success"] is True
        mock_post.assert_called_once()

    @patch("rival_ai.core.agent_definition.requests.post")
    def test_save_to_project_failure(self, mock_post):
        """Test failed save to project."""
        mock_post.side_effect = Exception("Network error")

        agent_def = AgentDefinition(
            name="Test Agent",
            description="A test agent",
            agent_object=Mock(),
            agent_type="langgraph",
            input_schema={"input": "string"},
            output_schema={"output": "string"},
        )

        with pytest.raises(APIError):
            agent_def.save_to_project("test-api-key", "test-project")
