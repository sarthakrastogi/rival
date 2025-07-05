"""Tests for attack criteria."""

import pytest

from rival_ai.generators.attack_criteria import (
    SubAttack,
    PromptInjectionAttack,
    get_attack_criteria,
)


class TestSubAttack:
    """Test cases for SubAttack class."""

    def test_subattack_creation(self):
        """Test creating a sub-attack."""
        sub_attack = SubAttack(
            id="test_attack",
            name="Test Attack",
            description="A test attack",
            example_input="test input",
        )

        assert sub_attack.id == "test_attack"
        assert sub_attack.name == "Test Attack"

    def test_format_to_string(self):
        """Test formatting sub-attack to string."""
        sub_attack = SubAttack(
            id="test_attack",
            name="Test Attack",
            description="A test attack",
            example_input="test input",
        )

        formatted = sub_attack.format_to_string()
        assert "test_attack" in formatted
        assert "Test Attack" in formatted
        assert "A test attack" in formatted
        assert "test input" in formatted


class TestPromptInjectionAttack:
    """Test cases for PromptInjectionAttack class."""

    def test_prompt_injection_creation(self):
        """Test creating prompt injection attack criteria."""
        criteria = PromptInjectionAttack(agent_definition="test agent")

        assert criteria.agent_definition == "test agent"
        assert len(criteria.sub_attacks) == 5
        assert any(sub.id == "direct_command" for sub in criteria.sub_attacks)

    def test_get_generation_messages(self):
        """Test generating messages for LLM."""
        criteria = PromptInjectionAttack(agent_definition="test agent")

        messages = criteria.get_generation_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test agent"

    def test_get_full_attack_definition(self):
        """Test getting full attack definition."""
        criteria = PromptInjectionAttack(agent_definition="test agent")

        full_def = criteria.get_full_attack_definition()
        assert "injection techniques" in full_def
        assert "Sub-attack techniques:" in full_def


class TestAttackCriteriaRegistry:
    """Test cases for attack criteria registry."""

    def test_get_attack_criteria_success(self):
        """Test getting attack criteria by name."""
        criteria_class = get_attack_criteria("prompt_injection")
        assert criteria_class == PromptInjectionAttack

    def test_get_attack_criteria_unknown(self):
        """Test getting unknown attack criteria."""
        with pytest.raises(ValueError):
            get_attack_criteria("unknown_attack")
