from abc import ABC, abstractmethod

from ingress.domain.ingress_plan import IngressPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class PlanStore(ABC):
    @abstractmethod
    async def get_plan_version_by_flow_id(self, flow_id: FlowId) -> PlanVersion: ...

    @abstractmethod
    async def put_plan(self, plan: IngressPlan) -> None: ...

    @abstractmethod
    async def get_plan(self, plan_version: PlanVersion) -> IngressPlan: ...
