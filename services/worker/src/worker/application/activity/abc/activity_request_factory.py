from abc import ABC, abstractmethod

from worker.domain.activity.activity_request import ActivityRequest, ActivityType


class ActivityRequestFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_request_for_activity_type(
        activity_type: ActivityType,
    ) -> ActivityRequest: ...
