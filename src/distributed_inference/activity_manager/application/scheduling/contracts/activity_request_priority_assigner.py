from abc import ABC, abstractmethod

from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
)


class ActivityRequestPriorityAssigner(ABC):
    @abstractmethod
    def compute_priority_for_activity_request(
        self, activity_request: ActivityRequest
    ) -> int: ...
