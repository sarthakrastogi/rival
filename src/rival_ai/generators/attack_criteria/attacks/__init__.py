"""Attack implementations subpackage.

This subpackage contains specific attack criteria implementations.
Each attack type is defined in its own module for better organization.
"""

from .harmful_content_generation import (
    HarmfulContentRequests,
    BiasAmplification,
    MisinformationGeneration,
    ViolenceAndHarmAdvocacy,
)
from .privacy_and_data_security import (
    PersonalInformationExtraction,
    DataLeakageScenarios,
    PrivacyBoundaryTesting,
)
from .prompt_manipulation_and_instruction_adherence import (
    PromptInjectionAttack,
    SystemPromptExtraction,
    InstructionHierarchyConfusion,
    DelimiterAndFormatExploitation,
)


# Export all attack implementations
__all__ = [
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
