import asyncio
from typing import override

from distributed_inference.activity_manager.application.ports.inbound.activity_manager import (
    ActivityManager,
)
from distributed_inference.activity_manager.application.ports.outbound.resource_manager import (
    ResourceManager,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityGrantInfo,
    ActivityRequest,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceLease,
)
from distributed_inference.activity_manager.application.scheduling.contracts.activity_request_scheduler import (
    ActivityRequestScheduler,
)
from distributed_inference.building_blocks.lifecycle.async_lifecycle import (
    AsyncLifecycle,
)


class DefaultActivityManager(AsyncLifecycle, ActivityManager):
    def __init__(
        self,
        request_scheduler: ActivityRequestScheduler,
        resource_manager: ResourceManager,
    ) -> None:
        super().__init__()
        self._request_scheduler = request_scheduler
        self._resource_manager = resource_manager

        ## TODO Maibe we should add this as an outbound port
        ## And implement local management as an outbound adapter
        self._pending_activity_grants: dict[ActivityGrantId, ResourceLease] = {}

    @override
    async def get_activity_grant_info(
        self, request: ActivityRequest
    ) -> ActivityGrantInfo:

        activity_grant_info_future: asyncio.Future[ActivityGrantInfo] = (
            asyncio.get_running_loop().create_future()
        )
        await self._request_scheduler.enqueue(request, activity_grant_info_future)

        activity_grant_info = await activity_grant_info_future
        return activity_grant_info

    @override
    async def release_activity_grant(self, activity_grant_id: ActivityGrantId) -> None:

        resource_lease = self._pending_activity_grants.pop(activity_grant_id, None)
        if resource_lease is None:
            return

        await self._resource_manager.release_resource_lease(
            resource_lease.resource_lease_id
        )

    @override
    async def renew_activity_grant(
        self, activity_grant_id: ActivityGrantId
    ) -> ActivityGrant:
        raise NotImplementedError

    @override
    async def start(self) -> None:
        while True:
            activity_request, future = await self._request_scheduler.dequeue()

            resource_lease = await self._resource_manager.acquire_resource_lease(
                activity_request.activity_resources
            )
            activity_grant_id = ActivityGrantId()
            self._pending_activity_grants[activity_grant_id] = resource_lease
            activity_grant = ActivityGrantInfo(activity_grant_id=activity_grant_id)
            future.set_result(activity_grant)

    @override
    async def stop(self) -> None:
        raise NotImplementedError
