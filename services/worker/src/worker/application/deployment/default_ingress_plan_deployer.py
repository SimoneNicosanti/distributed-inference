from typing import override

from shared.plan.plan_version import PlanVersion
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.application.ports.outbound.plan_store.ingress_plan_store import (
    IngressPlanStore,
)
from worker.domain.plan.ingress.ingress_plan import IngressPlan


class DefaultIngressPlanDeployer(PlanDeployer[IngressPlan]):
    def __init__(
        self,
        ingress_plan_store: IngressPlanStore,
    ) -> None:
        self._ingress_plan_store = ingress_plan_store
        super().__init__()

    @override
    async def prepare_plan(self, plan: IngressPlan) -> None:
        await self._ingress_plan_store.put_ingress_plan(plan)

    @override
    async def apply_plan(self, plan_version: PlanVersion) -> None:
        await self._ingress_plan_store.activate_ingress_plan(plan_version)
