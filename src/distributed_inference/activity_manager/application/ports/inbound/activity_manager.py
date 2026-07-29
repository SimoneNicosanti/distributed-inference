from abc import ABC, abstractmethod

from distributed_inference.activity_manager.domain.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityGrantInfo,
    ActivityRequest,
)


class ActivityManager(ABC):
    async def request_activity_grant(
        self, activity_request: ActivityRequest
    ) -> ActivityGrant:

        activity_grant_info = await self.get_activity_grant_info(activity_request)
        return ActivityGrant(activity_grant_info, self.release_activity_grant)

    @abstractmethod
    async def get_activity_grant_info(
        self,
        request: ActivityRequest,
    ) -> ActivityGrantInfo: ...

    @abstractmethod
    async def release_activity_grant(
        self, activity_grant_id: ActivityGrantId
    ) -> None: ...

    @abstractmethod
    async def renew_activity_grant(
        self, activity_grant_id: ActivityGrantId
    ) -> ActivityGrant: ...
