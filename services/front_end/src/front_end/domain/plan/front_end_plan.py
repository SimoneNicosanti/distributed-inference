from dataclasses import dataclass

from front_end.domain.plan.flow_plan import FlowPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class FrontEndPlan:
    plan_version: PlanVersion

    flow_ids: tuple[FlowId, ...]
    flow_plans: tuple[FlowPlan, ...]

    def __post_init__(self) -> None:
        if len(self.flow_ids) == 0:
            raise ValueError("Front end plan must not be empty")

        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Flow ids must be unique")

        if len(self.flow_plans) != len(self.flow_ids):
            raise ValueError("Each flow must have a flow plan")

    def get_flow_plan_by_flow_id(self, flow_id: FlowId) -> FlowPlan:
        try:
            flow_index = self.flow_ids.index(flow_id)
        except ValueError:
            raise ValueError(f"Flow {flow_id} not found in ingress plan") from None

        return self.flow_plans[flow_index]
