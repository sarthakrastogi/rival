"""Tests for test case generator."""

import pytest
from unittest.mock import Mock

from rival_ai.generators.testcase_generator import TestCaseGenerator
from rival_ai.generators.attack_criteria import PromptInjectionAttack
from rival_ai.exceptions import TestCaseGenerationError


class TestTestCaseGenerator:
    """Test cases for TestCaseGenerator class."""

    def test_generator_creation(self):
        """Test creating a test case generator."""
        generator = TestCaseGenerator(model="test-model")
        assert generator.model == "test-model"

    def test_generate_local_success(self, mock_agent_definition, mock_llm_callable):
        """Test successful local generation."""
        generator = TestCaseGenerator()

        testcases = list(
            generator.generate_local(
                agent_definition=mock_agent_definition,
                attack_types=["prompt_injection"],
                llm_callable=mock_llm_callable,
            )
        )

        assert len(testcases) == 1
        assert testcases[0].id == "direct_command"
        assert "harmful" in testcases[0].inputs["message"]

    def test_generate_local_no_llm_callable(self, mock_agent_definition):
        """Test local generation without LLM callable."""
        generator = TestCaseGenerator()

        with pytest.raises(TestCaseGenerationError):
            list(
                generator.generate_local(
                    agent_definition=mock_agent_definition,
                    attack_types=["prompt_injection"],
                )
            )

    def test_generate_with_criteria(self, mock_agent_definition, mock_llm_callable):
        """Test generation with custom criteria."""
        generator = TestCaseGenerator()
        criteria = PromptInjectionAttack(agent_definition="test agent")

        testcases = generator.generate_with_criteria(
            agent_definition=mock_agent_definition,
            attack_criteria=criteria,
            llm_callable=mock_llm_callable,
        )

        assert len(testcases) == 1
        assert testcases[0].id == "direct_command"

    def test_parse_json_response_with_code_block(self):
        """Test parsing JSON response with code block."""
        generator = TestCaseGenerator()
        response = """```json
[{"id": "test", "inputs": {}, "pass_criteria": []}]
```"""

        result = generator._parse_json_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "test"

    def test_parse_json_response_plain_json(self):
        """Test parsing plain JSON response."""
        generator = TestCaseGenerator()
        response = '[{"id": "test", "inputs": {}, "pass_criteria": []}]'

        result = generator._parse_json_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "test"

    def test_parse_json_response_single_object(self):
        """Test parsing single JSON object response."""
        generator = TestCaseGenerator()
        response = '{"id": "test", "inputs": {}, "pass_criteria": []}'

        result = generator._parse_json_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "test"
