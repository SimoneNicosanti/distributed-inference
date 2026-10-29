from typing import Protocol, override

from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.plan_store.plan_store_reader import (
    PlanStoreReader,
)
from worker.application.ports.outbound.plan_store.plan_store_writer import (
    PlanStoreWriter,
)


class VersionedPlan(Protocol):
    @property
    def plan_version(self) -> PlanVersion: ...


class InMemoryPlanStore[PlanT: VersionedPlan](
    PlanStoreWriter[PlanT],
    PlanStoreReader[PlanT],
):
    def __init__(self) -> None:
        self._active_version: PlanVersion | None = None
        self._plans_by_version: dict[PlanVersion, PlanT] = {}

    @override
    async def put_plan(self, plan: PlanT) -> None:
        existing = self._plans_by_version.get(plan.plan_version)
        if existing is not None and existing != plan:
            raise ValueError(f"Plan version {plan.plan_version} already exists")

        self._plans_by_version[plan.plan_version] = plan

    @override
    async def get_plan_by_version(
        self,
        version: PlanVersion,
    ) -> PlanT:
        if version in self._plans_by_version:
            return self._plans_by_version[version]
        raise KeyError(f"No plan found for plan version {version}")

    @override
    async def activate_plan(self, version: PlanVersion) -> None:
        if version not in self._plans_by_version:
            raise ValueError(f"Plan version {version} does not exist")

        self._active_version = version

    @override
    async def get_active_plan_version(self) -> PlanVersion:
        if self._active_version is None:
            raise ValueError("No active plan found")
        return self._active_version
