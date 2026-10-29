from dataclasses import dataclass

import pytest

from shared.plan.plan_version import PlanVersion
from worker.adapters.outbound.plan_store.in_memory_plan_store import InMemoryPlanStore


@dataclass(frozen=True, slots=True)
class _Plan:
    plan_version: PlanVersion
    value: str


@pytest.mark.asyncio
async def test_stores_reads_and_activates_a_versioned_plan() -> None:
    store = InMemoryPlanStore[_Plan]()
    plan = _Plan(PlanVersion(version_number=1), "one")

    await store.put_plan(plan)
    await store.activate_plan(plan.plan_version)

    assert await store.get_plan_by_version(plan.plan_version) == plan
    assert await store.get_active_plan_version() == plan.plan_version


@pytest.mark.asyncio
async def test_rejects_a_different_plan_with_the_same_version() -> None:
    store = InMemoryPlanStore[_Plan]()
    version = PlanVersion(version_number=1)
    await store.put_plan(_Plan(version, "one"))

    with pytest.raises(ValueError, match="already exists"):
        await store.put_plan(_Plan(version, "different"))
