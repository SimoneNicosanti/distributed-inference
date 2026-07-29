from typing import override

from distributed_inference.activity_manager.application.ports.inbound.activity_manager import (
    ActivityManager,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityGrantInfo,
    ActivityRequest,
)


class LocalActivityManagerAdapter(ActivityManager):
    def __init__(self, activity_manager: ActivityManager) -> None:
        super().__init__()
        self._activity_manager = activity_manager

    @override
    async def get_activity_grant_info(
        self,
        request: ActivityRequest,
    ) -> ActivityGrantInfo:
        return await self._activity_manager.get_activity_grant_info(request)

    @override
    async def release_activity_grant(self, activity_grant_id: ActivityGrantId) -> None:
        await self._activity_manager.release_activity_grant(activity_grant_id)

    @override
    async def renew_activity_grant(
        self, activity_grant_id: ActivityGrantId
    ) -> ActivityGrant:
        return await self._activity_manager.renew_activity_grant(activity_grant_id)

    pass
