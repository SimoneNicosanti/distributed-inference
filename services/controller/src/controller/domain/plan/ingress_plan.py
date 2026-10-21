from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from controller.domain.plan.flow_plan import FlowPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class IngressPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: PlanVersion
    flow_ids: tuple[FlowId, ...]
    flow_plans: tuple[FlowPlan, ...]

    @model_validator(mode="after")
    def validate_flows(self) -> Self:
        if not self.flow_ids:
            raise ValueError("Ingress plan must not be empty")
        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Flow ids must be unique")
        if len(self.flow_ids) != len(self.flow_plans):
            raise ValueError("Each flow must have a flow plan")
        return self
