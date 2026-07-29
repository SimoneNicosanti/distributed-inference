from dataclasses import dataclass
from enum import StrEnum, auto
from types import TracebackType
from typing import Awaitable, Callable, Self
from uuid import UUID, uuid4


class ResourceType(StrEnum):
    COMPUTE = auto()
    MEMORY = auto()
    NETWORK = auto()


@dataclass(frozen=True)
class LockRequirement:
    quantity: float
    exclusive: bool

    def __post_init__(self) -> None:

        if self.exclusive:
            if self.quantity != 0:
                raise ValueError(
                    "An exclusive resource requirement must have quantity equal to 0"
                )
        else:
            if self.quantity <= 0:
                raise ValueError("Quantity must be positive")


type ResourceAvailability = dict[ResourceType, float]

type ResourceLock = dict[ResourceType, LockRequirement]


@dataclass(frozen=True)
class ResourceLeaseId:
    id: UUID

    @classmethod
    def generate(cls) -> Self:
        return cls(uuid4())


class ResourceLease:
    type ReleaseCallback = Callable[[ResourceLeaseId], Awaitable[None]]

    def __init__(
        self,
        resource_lease_id: ResourceLeaseId,
        release_callback: ReleaseCallback,
    ) -> None:
        self._resource_lease_id = resource_lease_id
        self._release_callback = release_callback
        self._released = False

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
            await self._release_callback(self._resource_lease_id)
        except BaseException:
            self._released = False
            raise
