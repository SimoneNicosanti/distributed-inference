import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum, auto
from types import MappingProxyType, TracebackType
from typing import Self
from uuid import UUID, uuid4


class ResourceType(StrEnum):
    COMPUTE = auto()
    NETWORK = auto()


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    quantity: float
    exclusive: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantity):
            raise ValueError("Resource quantity must be finite")

        if self.exclusive:
            if self.quantity != 0:
                raise ValueError(
                    "An exclusive resource requirement must have quantity equal to 0"
                )
        else:
            if self.quantity <= 0:
                raise ValueError("Quantity must be positive")


class ActivityType(StrEnum):
    EXECUTION_INFERENCE = auto()
    EXECUTION_PROFILING = auto()

    NETWORK_TRANSMISSION = auto()
    NETWORK_PROFILING = auto()


@dataclass(frozen=True, slots=True)
class ActivityRequest:
    activity_type: ActivityType
    resource_requirements: Mapping[ResourceType, ResourceRequirement]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_requirements",
            MappingProxyType(dict(self.resource_requirements)),
        )


@dataclass(frozen=True, slots=True)
class ActivityGrantId:
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class ActivityGrantInfo:
    activity_grant_id: ActivityGrantId


class ActivityGrant:
    type ReleaseCallback = Callable[[ActivityGrantId], Awaitable[None]]

    def __init__(
        self, activity_grant_info: ActivityGrantInfo, release_callback: ReleaseCallback
    ) -> None:
        self.activity_grant_info = activity_grant_info

        self._released = False
        self._release_callback = release_callback

    async def __aenter__(self) -> Self:
        if self._released:
            raise RuntimeError("Activity grant already released")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return

        self._released = True

        try:
            await self._release_callback(self.activity_grant_info.activity_grant_id)
        except BaseException:
            self._released = False
            raise
