"""
Rival AI - A library for testing AI agent safety and security.
"""

from .core.benchmarking import Benchmarking
from .core.agent_definition import AgentDefinition
from .core.testcase import AttackTestcase
from .core.evaluation import EvaluationResult
from .core.benchmarking_result import BenchmarkingResult
from .generators.testcase_generator import TestCaseGenerator
from .detectors.narasimha.detector import NarasimhaAttackDetector

__version__ = "0.1.0"
__all__ = [
    "Benchmarking",
    "AgentDefinition",
    "AttackTestcase",
    "EvaluationResult",
    "BenchmarkingResult",
    "TestCaseGenerator",
    "NarasimhaAttackDetector",
]
