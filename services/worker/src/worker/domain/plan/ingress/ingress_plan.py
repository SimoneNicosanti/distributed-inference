from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from worker.domain.plan.ingress.flow_plan import FlowPlan


class IngressPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: PlanVersion

    flow_ids: tuple[FlowId, ...]
    flow_plans: tuple[FlowPlan, ...]

    @model_validator(mode="after")
    def validate_flow_plan(self) -> Self:
        if len(self.flow_ids) == 0:
            raise ValueError("Ingress plan must not be empty")

        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Flow ids must be unique")

        if len(self.flow_plans) != len(self.flow_ids):
            raise ValueError("Each flow must have a flow plan")

        return self

    def get_flow_plan_by_flow_id(self, flow_id: FlowId) -> FlowPlan:
        try:
            flow_index = self.flow_ids.index(flow_id)
        except ValueError:
            raise ValueError(f"Flow {flow_id} not found in ingress plan") from None

        return self.flow_plans[flow_index]
