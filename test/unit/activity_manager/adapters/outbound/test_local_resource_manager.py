import pytest

from distributed_inference.activity_manager.adapters.outbound.local_resource_manager import (
    LocalResourceManager,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceLeaseId,
    ResourceRequirement,
    ResourceType,
)


@pytest.mark.unit
def test_constructor_copies_and_validates_resource_availability() -> None:
    availability = {ResourceType.COMPUTE: 4.0}
    manager = LocalResourceManager(availability)

    availability[ResourceType.COMPUTE] = 1.0

    assert manager._all_resources == {ResourceType.COMPUTE: 4.0}
    assert manager._free_resources == {ResourceType.COMPUTE: 4.0}

    with pytest.raises(ValueError, match="negative quantity"):
        LocalResourceManager({ResourceType.COMPUTE: -1.0})


@pytest.mark.unit
@pytest.mark.parametrize(
    "quantity",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_constructor_rejects_non_finite_availability(quantity: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        LocalResourceManager({ResourceType.COMPUTE: quantity})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquire_rejects_unknown_and_oversized_resources() -> None:
    manager = LocalResourceManager({ResourceType.COMPUTE: 4.0})

    with pytest.raises(ValueError, match="Unknown resource type"):
        await manager.acquire_resource_lease(
            {
                ResourceType.MEMORY: ResourceRequirement(
                    quantity=1,
                    exclusive=False,
                )
            }
        )

    with pytest.raises(ValueError, match="exceeds"):
        await manager.acquire_resource_lease(
            {
                ResourceType.COMPUTE: ResourceRequirement(
                    quantity=5,
                    exclusive=False,
                )
            }
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_releasing_unknown_lease_is_idempotent() -> None:
    manager = LocalResourceManager({ResourceType.COMPUTE: 4.0})

    await manager.release_resource_lease(ResourceLeaseId())
    await manager.release_resource_lease(ResourceLeaseId())

    assert manager._free_resources == {ResourceType.COMPUTE: 4.0}
    assert manager._current_leases == {}
