import math
from collections.abc import Awaitable, Callable
from enum import StrEnum, auto
from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResourceType(StrEnum):
    COMPUTE = auto()
    MEMORY = auto()
    NETWORK = auto()


class ResourceRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)
    quantity: float
    exclusive: bool

    @model_validator(mode="after")
    def validate_request(self) -> Self:

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

        return self


type ResourceLock = dict[ResourceType, ResourceRequirement]


class ActivityType(StrEnum):
    INFERENCE_EXECUTION = auto()
    INFERENCE_FORWARDING = auto()

    PROFILING_NETWORK = auto()
    PROFILING_EXECUTION = auto()


class ActivityRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    activity_type: ActivityType
    required_locks: ResourceLock


class ActivityGrantId(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)


class ActivityGrantInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
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
