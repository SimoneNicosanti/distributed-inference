from typing import override

from distributed_inference.application.lifecycle.contracts.async_lifecycle import (
    AsyncLifecycle,
)
from distributed_inference.application.worker.contracts.activity.activity_request import (
    ActivityResponse,
)
from distributed_inference.application.worker.contracts.activity.activity_request_scheduler import (
    ActivityRequestScheduler,
)


class DefaultActivityCoordinator(AsyncLifecycle):
    def __init__(self, request_scheduler: ActivityRequestScheduler) -> None:
        super().__init__()
        self._request_scheduler = request_scheduler

    @override
    async def start(self) -> None:
        while True:
            _, future = await self._request_scheduler.dequeue()
            activity_response = ActivityResponse(True)
            future.set_result(activity_response)

    @override
    async def stop(self) -> None:
        pass
