from abc import ABC, abstractmethod
from typing import Any


class StaticPriorityAssigner(ABC):
    @abstractmethod
    def assign_priority(self, request: Any) -> int:
        raise NotImplementedError
