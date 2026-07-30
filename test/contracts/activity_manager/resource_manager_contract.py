import asyncio
from abc import ABC, abstractmethod

import pytest

from distributed_inference.activity_manager.application.ports.outbound.resource_manager import (
    ResourceManager,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceAvailability,
    ResourceRequirement,
    ResourceType,
)


class ResourceManagerContract(ABC):
    @abstractmethod
    def build_resource_manager(
        self,
        resource_availability: ResourceAvailability,
    ) -> ResourceManager:
        raise NotImplementedError

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_released_shared_resources_can_be_acquired_again(self) -> None:
        manager = self.build_resource_manager({ResourceType.COMPUTE: 4})
        lock = {
            ResourceType.COMPUTE: ResourceRequirement(
                quantity=4,
                exclusive=False,
            )
        }

        first_lease = await manager.acquire_resource_lease(lock)
        await manager.release_resource_lease(first_lease.resource_lease_id)
        second_lease = await manager.acquire_resource_lease(lock)

        assert second_lease.resource_lease_id != first_lease.resource_lease_id

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_shared_request_waits_until_capacity_is_released(self) -> None:
        manager = self.build_resource_manager({ResourceType.COMPUTE: 4})
        first_lease = await manager.acquire_resource_lease(
            {
                ResourceType.COMPUTE: ResourceRequirement(
                    quantity=3,
                    exclusive=False,
                )
            }
        )

        waiting_lease = asyncio.create_task(
            manager.acquire_resource_lease(
                {
                    ResourceType.COMPUTE: ResourceRequirement(
                        quantity=2,
                        exclusive=False,
                    )
                }
            )
        )
        await asyncio.sleep(0)
        assert not waiting_lease.done()

        await manager.release_resource_lease(first_lease.resource_lease_id)
        acquired_lease = await asyncio.wait_for(waiting_lease, timeout=1.0)

        assert acquired_lease.resource_lease_id != first_lease.resource_lease_id

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_exclusive_and_shared_leases_block_each_other(self) -> None:
        manager = self.build_resource_manager({ResourceType.COMPUTE: 4})
        shared_lease = await manager.acquire_resource_lease(
            {
                ResourceType.COMPUTE: ResourceRequirement(
                    quantity=1,
                    exclusive=False,
                )
            }
        )
        exclusive_waiter = asyncio.create_task(
            manager.acquire_resource_lease(
                {
                    ResourceType.COMPUTE: ResourceRequirement(
                        quantity=0,
                        exclusive=True,
                    )
                }
            )
        )
        await asyncio.sleep(0)
        assert not exclusive_waiter.done()

        await manager.release_resource_lease(shared_lease.resource_lease_id)
        exclusive_lease = await asyncio.wait_for(exclusive_waiter, timeout=1.0)

        shared_waiter = asyncio.create_task(
            manager.acquire_resource_lease(
                {
                    ResourceType.COMPUTE: ResourceRequirement(
                        quantity=1,
                        exclusive=False,
                    )
                }
            )
        )
        await asyncio.sleep(0)
        assert not shared_waiter.done()

        await manager.release_resource_lease(exclusive_lease.resource_lease_id)
        acquired_shared_lease = await asyncio.wait_for(shared_waiter, timeout=1.0)

        assert acquired_shared_lease.resource_lease_id != shared_lease.resource_lease_id
