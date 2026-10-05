import asyncio
from typing import override

from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.domain.activity.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityGrantInfo,
    ActivityRequest,
    ResourceType,
)


class LocalActivityManagerAdapter(ActivityManager):
    def __init__(
        self,
        available_resources: dict[ResourceType, float],
    ) -> None:
        super().__init__()
        self._condition = asyncio.Condition()
        self._all_res = available_resources.copy()
        self._current_available_res: dict[ResourceType, float] = (
            available_resources.copy()
        )
        # self._semaphores: dict[ResourceType, sysv_ipc.Semaphore] = {}
        self._grants: dict[ActivityGrantId, dict[ResourceType, float]] = {}

    @override
    async def acquire_activity_grant(
        self,
        request: ActivityRequest,
    ) -> ActivityGrant:

        required_allocation = {
            resource_type: self._all_res[resource_type]
            if requirement.exclusive
            else requirement.quantity
            for resource_type, requirement in request.resource_requirements.items()
        }

        async with self._condition:
            await self._condition.wait_for(
                lambda: all(
                    self._current_available_res[resource_type] >= required_quantity
                    for resource_type, required_quantity in required_allocation.items()
                )
            )

            for resource_type, requirement in required_allocation.items():
                self._current_available_res[resource_type] -= requirement

            grant_id = ActivityGrantId()
            self._grants[grant_id] = required_allocation

        return ActivityGrant(
            activity_grant_info=ActivityGrantInfo(activity_grant_id=grant_id),
            release_callback=self.release_activity_grant,
        )

    @override
    async def release_activity_grant(
        self,
        activity_grant_id: ActivityGrantId,
    ) -> None:
        async with self._condition:
            allocation = self._grants.pop(activity_grant_id)
            for resource_type, requirement in allocation.items():
                self._current_available_res[resource_type] += requirement

            self._condition.notify_all()
