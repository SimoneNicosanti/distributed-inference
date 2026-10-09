from functools import total_ordering
from typing import Any

from pydantic import BaseModel, ConfigDict, PositiveInt


@total_ordering
class PlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    version_number: PositiveInt

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, PlanVersion):
            return NotImplemented
        return self.version_number < other.version_number
