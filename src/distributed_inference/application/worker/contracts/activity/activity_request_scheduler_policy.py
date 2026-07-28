from abc import ABC, abstractmethod

from distributed_inference.application.worker.contracts.activity.activity_request import (
    ActivityRequest,
    ActivityType,
)


class ActivitySchedulerPolicy(ABC):
    @abstractmethod
    def can_start(
        self,
        request: ActivityRequest,
        current_activities: list[ActivityType],
        pending_activities: list[ActivityType],
    ) -> bool: ...
