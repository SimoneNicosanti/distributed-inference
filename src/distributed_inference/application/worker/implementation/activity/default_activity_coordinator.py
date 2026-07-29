from typing import override

from distributed_inference.application.lifecycle.contracts.async_lifecycle import (
    AsyncLifecycle,
)
from distributed_inference.application.worker.contracts.activity.activity_request import (
    ActivityGrant,
)
from distributed_inference.application.worker.contracts.activity.activity_request_scheduler import (
    ActivityRequestScheduler,
)
from distributed_inference.application.worker.contracts.resource.resource_manager import (
    ResourceManager,
)


class DefaultActivityCoordinator(AsyncLifecycle):
    def __init__(
        self,
        request_scheduler: ActivityRequestScheduler,
        resource_manager: ResourceManager,
    ) -> None:
        super().__init__()
        self._request_scheduler = request_scheduler
        self._resource_manager = resource_manager

    @override
    async def start(self) -> None:
        while True:
            activity_request, future = await self._request_scheduler.dequeue()

            resource_lease = await self._resource_manager.acquire_resource_lease(
                activity_request.resource_lock
            )
            activity_grant = ActivityGrant(resource_lease)
            future.set_result(activity_grant)

    @override
    async def stop(self) -> None:
        pass
