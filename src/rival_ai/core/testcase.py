"""Test case models and functionality."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from .agent_definition import AgentDefinition
from .evaluation import EvaluationResult
from ..config import config
from ..exceptions import ValidationError


class AttackTestcase(BaseModel):
    """Represents a test case for attacking/testing an AI agent."""

    id: str = Field(..., description="Unique identifier for the test case")
    inputs: Dict[str, Any] = Field(..., description="Input parameters for the test")
    pass_criteria: List[str] = Field(
        ..., description="Criteria that must be met to pass"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the test case to a dictionary."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackTestcase":
        """
        Create an AttackTestcase from a dictionary.

        Args:
            data: Dictionary containing test case data

        Returns:
            AttackTestcase instance

        Raises:
            ValidationError: If the data is invalid
        """
        try:
            return cls(
                id=data["id"],
                inputs=data["inputs"],
                pass_criteria=data["pass_criteria"],
            )
        except KeyError as e:
            raise ValidationError(f"Missing required field: {e}")
        except Exception as e:
            raise ValidationError(f"Invalid test case data: {str(e)}")

    def evaluate_from_api(
        self,
        api_key: str,
        project_id: str,
        agent_definition: str,
        agent_internal_state: Dict[str, Any],
        agent_response: Dict[str, Any],
        model: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate this test case against an agent's response using the API.

        Args:
            api_key: API key for authentication
            project_id: Project ID
            agent_definition: String representation of the agent
            agent_internal_state: Agent's internal state during execution
            agent_response: Agent's final response
            model: Model to use for evaluation (optional)

        Returns:
            EvaluationResult with the evaluation outcome

        Raises:
            APIError: If the API request fails
        """
        raise NotImplementedError("API-based evaluation not implemented yet")

    def evaluate(
        self,
        agent_definition: AgentDefinition,
        agent_response: Any,
        agent_internal_state: Optional[Any] = None,
        model: Optional[str] = None,
        project_id: str = "",
    ) -> EvaluationResult:
        """
        Evaluate this test case locally using LLM calls.

        Args:
            agent_definition: String representation of the agent
            agent_internal_state: Agent's internal state during execution
            agent_response: Agent's final response
            model: Model to use for evaluation (optional)
            project_id: Project identifier (optional)

        Returns:
            EvaluationResult with the evaluation outcome
        """
        from ..evaluators import LocalEvaluator

        evaluator = LocalEvaluator(default_model=model or config.default_model)
        return evaluator.evaluate_testcase(
            project_id=project_id,
            testcase=self,
            agent_definition=agent_definition,
            agent_internal_state=agent_internal_state,
            agent_response=agent_response,
            model=model,
        )
