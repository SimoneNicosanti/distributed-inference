import asyncio
from copy import deepcopy
from typing import override

from distributed_inference.activity_manager.application.ports.outbound.resource_manager import (
    ResourceManager,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceAvailability,
    ResourceLease,
    ResourceLeaseId,
    ResourceLock,
    ResourceType,
)


class LocalResourceManager(ResourceManager):
    def __init__(self, resource_availability: ResourceAvailability) -> None:
        super().__init__()
        self._condition = asyncio.Condition()

        self._all_resources: ResourceAvailability = deepcopy(resource_availability)
        self._validate_all_resources(self._all_resources)

        self._free_resources: ResourceAvailability = deepcopy(resource_availability)
        self._exclusively_locked: set[ResourceType] = set()

        self._current_leases: dict[ResourceLeaseId, ResourceLock] = {}

    @override
    async def acquire_resource_lease(
        self, resource_lock: ResourceLock
    ) -> ResourceLease:
        lock_to_get = dict(resource_lock)  ## We copy allocated resources
        self._validate_resource_lock(lock_to_get)
        async with self._condition:
            await self._condition.wait_for(lambda: self._can_reserve(lock_to_get))
            self._reserve_all(lock_to_get)

            lease_id = ResourceLeaseId()
            self._current_leases[lease_id] = lock_to_get

        resource_lease = ResourceLease(
            resource_lease_id=lease_id,
        )
        return resource_lease

    @override
    async def release_resource_lease(self, resource_lease_id: ResourceLeaseId) -> None:
        async with self._condition:
            released_lock = self._current_leases.pop(resource_lease_id, None)
            if released_lock is None:
                return
            self._release_all(released_lock)
            self._condition.notify_all()

    def _can_reserve(self, resource_lock: ResourceLock) -> bool:
        for resource_type, requirement in resource_lock.items():
            if resource_type in self._exclusively_locked:
                return False

            if requirement.exclusive:
                if (
                    self._free_resources[resource_type]
                    != self._all_resources[resource_type]
                ):
                    return False
            elif self._free_resources[resource_type] < requirement.quantity:
                return False

        return True

    def _reserve_all(self, resources: ResourceLock) -> None:
        for resource_type, requirement in resources.items():
            if requirement.exclusive:
                self._exclusively_locked.add(resource_type)
            else:
                self._free_resources[resource_type] -= requirement.quantity

    def _release_all(self, released_lock: ResourceLock) -> None:
        for resource_type, requirement in released_lock.items():
            if requirement.exclusive:
                self._exclusively_locked.remove(resource_type)
            else:
                self._free_resources[resource_type] += requirement.quantity

    def _validate_resource_lock(self, resource_lock: ResourceLock) -> None:
        for resource_type, requirement in resource_lock.items():
            if resource_type not in self._all_resources:
                raise ValueError(f"Unknown resource type: {resource_type}")

            if (
                not requirement.exclusive
                and requirement.quantity > self._all_resources[resource_type]
            ):
                raise ValueError(
                    f"Requested quantity of {requirement.quantity} for {resource_type} exceeds "
                    f"the total available quantity {self._all_resources[resource_type]}"
                )

    def _validate_all_resources(self, all_resources: ResourceAvailability) -> None:
        import math

        for resource_type, quantity in all_resources.items():
            if not math.isfinite(quantity):
                raise ValueError("Resource quantity must be finite")
            if quantity < 0:
                raise ValueError(
                    f"Resource type {resource_type} has negative quantity {quantity}"
                )
