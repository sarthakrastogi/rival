"""Registry for all available attack criteria."""

from typing import Dict, Type, List
from .models import BaseAttackCriteria
from .attacks import (
    HarmfulContentRequests,
    BiasAmplification,
    MisinformationGeneration,
    ViolenceAndHarmAdvocacy,
    PersonalInformationExtraction,
    DataLeakageScenarios,
    PrivacyBoundaryTesting,
    PromptInjectionAttack,
    SystemPromptExtraction,
    InstructionHierarchyConfusion,
    DelimiterAndFormatExploitation,
)


class AttackCriteriaRegistry:
    """Registry for managing attack criteria types."""

    _registry: Dict[str, Type[BaseAttackCriteria]] = {
        "harmful_content_requests": HarmfulContentRequests,
        "bias_amplification": BiasAmplification,
        "misinformation_generation": MisinformationGeneration,
        "violence_and_harm_advocacy": ViolenceAndHarmAdvocacy,
        "personal_information_extraction": PersonalInformationExtraction,
        "data_leakage_scenarios": DataLeakageScenarios,
        "privacy_boundary_testing": PrivacyBoundaryTesting,
        "prompt_injection": PromptInjectionAttack,
        "system_prompt_extraction": SystemPromptExtraction,
        "instruction_hierarchy_confusion": InstructionHierarchyConfusion,
        "delimiter_and_format_exploitation": DelimiterAndFormatExploitation,
    }

    @classmethod
    def get_attack_criteria(cls, attack_type: str) -> Type[BaseAttackCriteria]:
        """Get an attack criteria class by name."""
        if attack_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown attack type: {attack_type}. "
                f"Available types: {available_types}"
            )
        return cls._registry[attack_type]

    @classmethod
    def register_attack_criteria(
        cls, name: str, attack_class: Type[BaseAttackCriteria]
    ):
        """Register a new attack criteria type."""
        cls._registry[name] = attack_class

    @classmethod
    def list_available_attacks(cls) -> List[str]:
        """List all available attack types."""
        return list(cls._registry.keys())

    @classmethod
    def get_all_attack_criteria(cls) -> Dict[str, Type[BaseAttackCriteria]]:
        """Get all registered attack criteria."""
        return cls._registry.copy()
