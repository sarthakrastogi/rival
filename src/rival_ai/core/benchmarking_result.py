"""Benchmarking result models and functionality."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .testcase import AttackTestcase


class BenchmarkingResult(BaseModel):
    """Complete result of benchmarking an agent against a test case."""

    project_id: str = Field(..., description="ID of the project")
    testcase: AttackTestcase = Field(
        ..., description="The test case that was evaluated"
    )
    agent_internal_state: Optional[Any] = Field(
        None, description="Agent's internal state during execution"
    )
    agent_final_output: Any = Field(..., description="Agent's final output")
    test_passes: List[int] = Field(
        ..., description="Pass/fail results (1=pass, 0=fail)"
    )
    evaluated_at: datetime = Field(..., description="When the evaluation was performed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()

    @property
    def passed(self) -> bool:
        """Check if the test case passed completely."""
        return all(result == 1 for result in self.test_passes)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate for this specific test."""
        if not self.test_passes:
            return 0.0
        return sum(self.test_passes) / len(self.test_passes)
