"""Tests for test case functionality."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from rival_ai.core.testcase import AttackTestcase
from rival_ai.core.evaluation import EvaluationResult
from rival_ai.exceptions import ValidationError, APIError


class TestAttackTestcase:
    """Test cases for AttackTestcase class."""

    def test_testcase_creation(self):
        """Test creating a test case."""
        testcase = AttackTestcase(
            id="test_1",
            inputs={"message": "test input"},
            pass_criteria=["criteria 1", "criteria 2"],
        )

        assert testcase.id == "test_1"
        assert testcase.inputs == {"message": "test input"}
        assert len(testcase.pass_criteria) == 2

    def test_to_dict(self):
        """Test converting test case to dictionary."""
        testcase = AttackTestcase(
            id="test_1", inputs={"message": "test input"}, pass_criteria=["criteria 1"]
        )

        result = testcase.to_dict()
        assert result["id"] == "test_1"
        assert result["inputs"] == {"message": "test input"}
        assert result["pass_criteria"] == ["criteria 1"]

    def test_from_dict_success(self):
        """Test creating test case from dictionary."""
        data = {
            "id": "test_1",
            "inputs": {"message": "test input"},
            "pass_criteria": ["criteria 1"],
        }

        testcase = AttackTestcase.from_dict(data)
        assert testcase.id == "test_1"
        assert testcase.inputs == {"message": "test input"}

    def test_from_dict_missing_field(self):
        """Test creating test case from invalid dictionary."""
        data = {
            "id": "test_1",
            "inputs": {"message": "test input"},
            # Missing pass_criteria
        }

        with pytest.raises(ValidationError):
            AttackTestcase.from_dict(data)

    @patch("rival_ai.core.testcase.requests.post")
    def test_evaluate_success(self, mock_post):
        """Test successful evaluation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "evaluation_result": {
                "test_passes": [1, 1, 0],
                "evaluated_at": datetime.now().isoformat(),
            },
        }
        mock_post.return_value = mock_response

        testcase = AttackTestcase(
            id="test_1", inputs={"message": "test input"}, pass_criteria=["criteria 1"]
        )

        result = testcase.evaluate(
            api_key="test-key",
            project_id="test-project",
            agent_definition="test agent",
            agent_internal_state={"state": "active"},
            agent_response={"response": "test response"},
        )

        assert isinstance(result, EvaluationResult)
        assert result.test_passes == [1, 1, 0]
        mock_post.assert_called_once()

    @patch("rival_ai.core.testcase.requests.post")
    def test_evaluate_failure(self, mock_post):
        """Test failed evaluation."""
        mock_post.side_effect = Exception("Network error")

        testcase = AttackTestcase(
            id="test_1", inputs={"message": "test input"}, pass_criteria=["criteria 1"]
        )

        with pytest.raises(APIError):
            testcase.evaluate(
                api_key="test-key",
                project_id="test-project",
                agent_definition="test agent",
                agent_internal_state={"state": "active"},
                agent_response={"response": "test response"},
            )
