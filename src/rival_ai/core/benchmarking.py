"""Benchmarking functionality for testing AI agents."""

import json
from datetime import datetime
from typing import List, Iterator, Optional
from pydantic import BaseModel, Field, validator
from uuid import uuid4, UUID

import requests

from .testcase import AttackTestcase
from .benchmarking_result import BenchmarkingResult
from ..config import config
from ..exceptions import APIError


class Benchmarking(BaseModel):
    """Main benchmarking class for running agent safety tests."""

    id: UUID = Field(default_factory=uuid4, exclude=True)
    project_id: str = Field(..., description="ID of the project")
    title: Optional[str] = Field(None, description="Title of the benchmarking run")
    run_datetime: datetime = Field(default_factory=datetime.utcnow)
    results: List[BenchmarkingResult] = Field(default_factory=list)

    @validator("title", pre=True, always=True)
    def set_default_title(cls, v, values):
        """Set default title if not provided."""
        if v is None:
            return f"Benchmarking on {datetime.utcnow().strftime('%Y-%m-%d')}"
        return v

    def to_dict(self) -> dict:
        """Convert the benchmarking instance to a dictionary."""
        return self.dict()

    def get_testcases_stream(
        self, api_key: str, limit: int = None
    ) -> Iterator[AttackTestcase]:
        """
        Stream test cases from the API.

        Args:
            api_key: API key for authentication
            limit: Maximum number of test cases to retrieve

        Yields:
            AttackTestcase: Individual test cases

        Raises:
            APIError: If the API request fails
        """
        if limit is None:
            limit = config.default_limit

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/x-ndjson",
        }

        params = {"project_id": self.project_id, "limit": limit}

        try:
            with requests.get(
                config.api_endpoints["testcases"],
                headers=headers,
                params=params,
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            raw_json = json.loads(line)
                            yield AttackTestcase.from_dict(raw_json)
                        except json.JSONDecodeError as e:
                            raise APIError(f"Invalid JSON in response: {str(e)}")

        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to fetch test cases: {str(e)}")

    def add_result(self, result: BenchmarkingResult) -> None:
        """Add a benchmarking result to the collection."""
        self.results.append(result)

    def get_summary(self) -> dict:
        """Get a summary of the benchmarking results."""
        if not self.results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_rate": 0.0,
            }

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if all(result.test_passes))
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": pass_rate,
        }

        print(f"\n--- Benchmarking Summary ---")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed tests: {summary['passed_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        print(f"Pass rate: {summary['pass_rate']:.2%}")

        return summary
