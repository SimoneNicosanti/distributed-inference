import asyncio

from shared.plan.plan_version import PlanVersion
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.service_plan import ServicePlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


class HttpServicePlanDeployer:
    def __init__(
        self,
        worker_plan_deployer: PlanDeployer[WorkerPlan],
        ingress_plan_deployer: PlanDeployer[IngressPlan] | None,
    ) -> None:
        self._worker_plan_deployer = worker_plan_deployer
        self._ingress_plan_deployer = ingress_plan_deployer

    async def prepare_service_plan(self, plan: ServicePlan) -> None:
        has_ingress_plan = plan.ingress_plan is not None
        has_ingress_deployer = self._ingress_plan_deployer is not None
        if has_ingress_plan != has_ingress_deployer:
            raise ValueError(
                "Service plan ingress component does not match service capabilities"
            )

        if plan.ingress_plan is not None and self._ingress_plan_deployer is not None:
            await asyncio.gather(
                self._worker_plan_deployer.prepare_plan(plan.worker_plan),
                self._ingress_plan_deployer.prepare_plan(plan.ingress_plan),
            )
        else:
            await self._worker_plan_deployer.prepare_plan(plan.worker_plan)

    async def apply_service_plan(self, plan_version: PlanVersion) -> None:
        if self._ingress_plan_deployer is not None:
            await asyncio.gather(
                self._worker_plan_deployer.apply_plan(plan_version),
                self._ingress_plan_deployer.apply_plan(plan_version),
            )
        else:
            await self._worker_plan_deployer.apply_plan(plan_version)
