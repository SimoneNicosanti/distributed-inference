from functools import total_ordering
from typing import Any

from pydantic import BaseModel, ConfigDict


@total_ordering
class InferencePlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    version_number: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, InferencePlanVersion):
            return NotImplemented
        return self.version_number < other.version_number


class ServiceInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: InferencePlanVersion


class WholeInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)
