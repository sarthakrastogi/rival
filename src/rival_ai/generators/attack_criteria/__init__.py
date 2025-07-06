"""Attack criteria package for test case generation.

This package provides base classes and specific attack criteria implementations
for generating test cases to probe AI agent safety boundaries.
"""

from .models import BaseAttackCriteria, SubAttack
from .registry import AttackCriteriaRegistry
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


# Export all public classes and functions
__all__ = [
    # Base classes
    "BaseAttackCriteria",
    "SubAttack",
    # Registry
    "AttackCriteriaRegistry",
    # Specific attack implementations
    "HarmfulContentRequests",
    "BiasAmplification",
    "MisinformationGeneration",
    "ViolenceAndHarmAdvocacy",
    "PersonalInformationExtraction",
    "DataLeakageScenarios",
    "PrivacyBoundaryTesting",
    "PromptInjectionAttack",
    "SystemPromptExtraction",
    "InstructionHierarchyConfusion",
    "DelimiterAndFormatExploitation",
]
