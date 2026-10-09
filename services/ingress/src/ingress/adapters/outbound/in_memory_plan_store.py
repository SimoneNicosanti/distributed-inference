from typing import override

from ingress.application.ports.outbound.plan_store import PlanStore
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class InMemoryPlanStore(PlanStore):
    @override
    async def get_plan_version_by_flow_id(self, flow_id: FlowId) -> PlanVersion:
        pass

    @override
    async def put_plan(self, )
