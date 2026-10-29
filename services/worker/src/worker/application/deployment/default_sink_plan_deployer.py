from typing import override

from shared.plan.plan_version import PlanVersion
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.application.ports.outbound.plan_store.plan_store_writer import (
    PlanStoreWriter,
)
from worker.domain.plan.sink.sink_plan import SinkPlan


class DefaultSinkPlanDeployer(PlanDeployer[SinkPlan]):
    def __init__(
        self,
        sink_plan_store: PlanStoreWriter[SinkPlan],
    ) -> None:
        self._sink_plan_store = sink_plan_store
        super().__init__()

    @override
    async def prepare_plan(self, plan: SinkPlan) -> None:
        await self._sink_plan_store.put_plan(plan)

    @override
    async def apply_plan(self, plan_version: PlanVersion) -> None:
        await self._sink_plan_store.activate_plan(plan_version)
