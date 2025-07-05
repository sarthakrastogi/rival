"""Evaluation result models and functionality."""

from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Result of evaluating a test case against an agent."""

    test_passes: List[int] = Field(
        ..., description="Pass/fail results (1=pass, 0=fail)"
    )
    evaluated_at: datetime = Field(..., description="When the evaluation was performed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()

    @property
    def passed(self) -> bool:
        """Check if all criteria passed."""
        return all(result == 1 for result in self.test_passes)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        if not self.test_passes:
            return 0.0
        return sum(self.test_passes) / len(self.test_passes)
