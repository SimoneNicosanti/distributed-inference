from dataclasses import dataclass

from controller.domain.plan.flow_plan import FlowPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class IngressPlan:
    plan_version: PlanVersion
    flow_ids: tuple[FlowId, ...]
    flow_plans: tuple[FlowPlan, ...]

    def __post_init__(self) -> None:
        if not self.flow_ids:
            raise ValueError("Ingress plan must not be empty")
        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Flow ids must be unique")
        if len(self.flow_ids) != len(self.flow_plans):
            raise ValueError("Each flow must have a flow plan")
