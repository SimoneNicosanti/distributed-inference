from abc import ABC, abstractmethod

from worker.domain.activity.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityRequest,
)


class ActivityManager(ABC):
    @abstractmethod
    async def acquire_activity_grant(
        self,
        request: ActivityRequest,
    ) -> ActivityGrant: ...

    @abstractmethod
    async def release_activity_grant(
        self, activity_grant_id: ActivityGrantId
    ) -> None: ...
