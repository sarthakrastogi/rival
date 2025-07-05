"""Local evaluation functionality using LLM calls."""

from typing import Any, List, Optional
from datetime import datetime

from ..core.agent_definition import AgentDefinition
from ..core.evaluation import EvaluationResult
from ..core.testcase import AttackTestcase
from ..exceptions import ValidationError
from ..utils import llm_call, parse_list_from_llm_response
from .prompts import AIAgentEvalPrompt
from ..config import config


class LocalEvaluator:
    """Local evaluator that uses LLM calls to evaluate test cases."""

    def __init__(self, default_model: str):
        """Initialize the local evaluator.

        Args:
            default_model: Default model to use for evaluation
        """
        self.default_model = default_model or config.default_model

    def evaluate_testcase(
        self,
        project_id: str,
        testcase: AttackTestcase,
        agent_definition: AgentDefinition,
        agent_response: Any,
        agent_internal_state: Optional[Any] = None,
        model: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a test case against an agent's response using local LLM calls.

        Args:
            project_id: Project identifier
            testcase: Test case to evaluate
            agent_definition: Representation of the agent
            agent_internal_state: Agent's internal state during execution
            agent_response: Agent's final response
            model: Model to use for evaluation (optional)

        Returns:
            EvaluationResult with the evaluation outcome

        Raises:
            ValidationError: If evaluation fails or returns invalid results
        """
        evaluation_model = model or self.default_model

        try:
            # Generate evaluation prompt
            prompt = AIAgentEvalPrompt(
                testcase=testcase,
                agent_definition=agent_definition.format_to_string(),
                agent_internal_state=str(agent_internal_state),
                agent_response=str(agent_response),
            )

            # Make LLM call
            llm_response = llm_call(messages=prompt.messages, model=evaluation_model)

            # Parse response as list
            test_passes: List[int] = parse_list_from_llm_response(llm_response)

            # Validate results
            if len(test_passes) != len(testcase.pass_criteria):
                raise ValidationError(
                    f"Expected {len(testcase.pass_criteria)} results, got {len(test_passes)}"
                )

            if not all(isinstance(x, int) and x in [0, 1] for x in test_passes):
                raise ValidationError("All results must be 0 or 1")

            # Create result
            eval_result = EvaluationResult(
                test_passes=test_passes,
                evaluated_at=datetime.now(),
            )
            print(f"Evaluation result: {eval_result}")
            print(f"Test passed: {eval_result.passed}")
            return eval_result

        except Exception as e:
            raise ValidationError(f"Evaluation failed: {str(e)}")

    def evaluate_testcase_simple(
        self,
        testcase: AttackTestcase,
        agent_definition: str,
        agent_response: Any,
        agent_internal_state: Optional[Any] = None,
        model: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Simplified evaluation that returns only the evaluation result.

        Args:
            testcase: Test case to evaluate
            agent_definition: String representation of the agent
            agent_internal_state: Agent's internal state during execution
            agent_response: Agent's final response
            model: Model to use for evaluation (optional)

        Returns:
            EvaluationResult with the evaluation outcome
        """
        result = self.evaluate_testcase(
            project_id="",  # Empty project ID for simple evaluation
            testcase=testcase,
            agent_definition=agent_definition,
            agent_internal_state=agent_internal_state,
            agent_response=agent_response,
            model=model,
        )
        return result.evaluation_result
