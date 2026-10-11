from typing import override

from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.plan_store.active_plan_version_reader import (
    ActivePlanVersionReader,
)
from worker.application.ports.outbound.plan_store.ingress_plan_store import (
    IngressPlanStore,
)
from worker.domain.plan.ingress.ingress_plan import IngressPlan


class InMemoryIngressPlanStore(IngressPlanStore, ActivePlanVersionReader):
    def __init__(self) -> None:
        self._latest_ingress_plan: IngressPlan | None = None
        self._active_ingress_plan: IngressPlan | None = None
        self._ingress_plans_by_version: dict[PlanVersion, IngressPlan] = {}

    @override
    async def put_ingress_plan(self, ingress_plan: IngressPlan) -> None:
        existing_plan = self._ingress_plans_by_version.get(ingress_plan.plan_version)
        if existing_plan is not None:
            if existing_plan != ingress_plan:
                raise ValueError(
                    f"Ingress plan version {ingress_plan.plan_version} already exists"
                )
            return

        if (
            self._latest_ingress_plan is None
            or ingress_plan.plan_version > self._latest_ingress_plan.plan_version
        ):
            self._latest_ingress_plan = ingress_plan

        self._ingress_plans_by_version[ingress_plan.plan_version] = ingress_plan

    @override
    async def get_ingress_plan_by_version(
        self, version: PlanVersion
    ) -> IngressPlan | None:
        return self._ingress_plans_by_version.get(version)

    @override
    async def activate_ingress_plan(self, version: PlanVersion) -> None:
        ingress_plan = self._ingress_plans_by_version.get(version)
        if ingress_plan is None:
            raise ValueError(f"Ingress plan version {version} does not exist")
        self._active_ingress_plan = ingress_plan

    @override
    async def get_active_plan_version(self) -> PlanVersion | None:
        if self._active_ingress_plan is None:
            return None
        return self._active_ingress_plan.plan_version
