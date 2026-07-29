from abc import ABC, abstractmethod

from distributed_inference.activity_manager.domain.resource_type import (
    ResourceLease,
    ResourceLeaseId,
    ResourceLock,
)


class ResourceManager(ABC):
    @abstractmethod
    async def acquire_resource_lease(
        self, resource_lock: ResourceLock
    ) -> ResourceLease: ...

    @abstractmethod
    async def release_resource_lease(
        self, resource_lease_id: ResourceLeaseId
    ) -> None: ...
