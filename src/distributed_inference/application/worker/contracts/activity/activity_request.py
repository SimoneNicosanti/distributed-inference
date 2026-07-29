from dataclasses import dataclass
from enum import StrEnum, auto
from types import TracebackType
from typing import Awaitable, Callable, Self

from distributed_inference.application.worker.contracts.resource.resource_type import (
    ResourceLease,
    ResourceLock,
)


class ActivityType(StrEnum):
    INFERENCE_EXECUTION = auto()
    INFERENCE_FORWARDING = auto()

    PROFILING_NETWORK = auto()
    PROFILING_EXECUTION = auto()


@dataclass
class ActivityRequest:
    activity_type: ActivityType
    resource_lock: ResourceLock


class ActivityGrant:
    type ReleaseCallback = Callable[[], Awaitable[None]]

    def __init__(self, resource_lease: ResourceLease) -> None:
        self._released = False
        self._resource_lease = resource_lease

    ## We define these two methods to allow context
    ## take and release with the with keyword
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.release()

    ## Once the grant usage has terminated, we can release the resources
    ## calling the release callback
    async def release(self) -> None:
        if self._released:
            return

        self._released = True

        try:
            await self._resource_lease.release()
        except BaseException:
            self._released = False
            raise
