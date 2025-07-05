"""Test case generation functionality."""

import json
from typing import List, Generator, Optional

from ..core.agent_definition import AgentDefinition
from ..core.testcase import AttackTestcase
from ..exceptions import TestCaseGenerationError
from .attack_criteria import (
    BaseAttackCriteria,
    ATTACK_CRITERIA_REGISTRY,
)
from ..utils import llm_call, parse_json_from_llm_response


class TestCaseGenerator:
    """Generator for creating attack test cases."""

    def __init__(self, model: str = None):
        """
        Initialize the test case generator.

        Args:
            model: LLM model to use for generation (optional)
        """
        from ..config import config

        self.model = model or config.default_model

    def generate_from_api(
        self,
        agent_definition: AgentDefinition,
        attack_types: Optional[List[str]] = None,
    ) -> Generator[AttackTestcase, None, None]:
        """
        Generate test cases using the API (original method).

        Args:
            agent_definition: The agent to generate test cases for
            attack_types: List of attack types to generate (optional)

        Yields:
            AttackTestcase: Generated test cases
        """
        # This would use the original API-based generation
        # Implementation would depend on the external API
        raise NotImplementedError("API-based generation not implemented yet")

    def generate_local(
        self,
        agent_definition: AgentDefinition,
        attack_types: Optional[List[str]] = None,
    ) -> Generator[AttackTestcase, None, None]:
        """
        Generate test cases locally using provided LLM function.

        Args:
            agent_definition: The agent to generate test cases for
            attack_types: List of attack types to generate (defaults to all)
            llm_callable: Function to call LLM (must accept messages and model params)

        Yields:
            AttackTestcase: Generated test cases

        Raises:
            TestCaseGenerationError: If generation fails
        """

        # Default to all attack types if none specified
        if attack_types is None:
            attack_types = list(ATTACK_CRITERIA_REGISTRY.keys())

        for attack_type in attack_types:
            try:
                # Get the attack criteria class
                attack_criteria_class = ATTACK_CRITERIA_REGISTRY[attack_type]

                # Create instance with agent definition
                attack_criteria = attack_criteria_class(
                    agent_definition=agent_definition
                )

                # Generate messages for LLM
                messages = attack_criteria.get_generation_messages()

                # Call LLM
                llm_response = llm_call(messages=messages, model=self.model)

                # Parse JSON response
                testcases_data = parse_json_from_llm_response(llm_response)

                # Convert to AttackTestcase objects
                for i, testcase_data in enumerate(testcases_data):
                    testcase = AttackTestcase.from_dict(testcase_data)
                    print(f"\n--- Test Case {i+1} ---")
                    print(f"ID: {testcase.id}")
                    print(f"Inputs: {testcase.inputs}")
                    print(f"Pass Criteria: {testcase.pass_criteria}")
                    yield testcase

            except Exception as e:
                raise TestCaseGenerationError(
                    f"Failed to generate test cases for attack type '{attack_type}': {str(e)}"
                )

    def generate_with_criteria(
        self,
        agent_definition: AgentDefinition,
        attack_criteria: BaseAttackCriteria,
    ) -> List[AttackTestcase]:
        """
        Generate test cases using custom attack criteria.

        Args:
            agent_definition: The agent to generate test cases for
            attack_criteria: Custom attack criteria to use
            llm_callable: Function to call LLM

        Returns:
            List of generated test cases

        Raises:
            TestCaseGenerationError: If generation fails
        """
        try:
            # Update agent definition in criteria
            attack_criteria.agent_definition = agent_definition.format_to_string()

            # Generate messages
            messages = attack_criteria.get_generation_messages()

            # Call LLM
            llm_response = llm_call(messages=messages, model=self.model)

            # Parse and return test cases
            testcases_data = parse_json_from_llm_response(llm_response)
            return [AttackTestcase.from_dict(data) for data in testcases_data]

        except Exception as e:
            raise TestCaseGenerationError(f"Failed to generate test cases: {str(e)}")
