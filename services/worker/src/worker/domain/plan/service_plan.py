from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


class ServicePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_plan: WorkerPlan
    ingress_plan: IngressPlan | None

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if (
            self.ingress_plan is not None
            and self.worker_plan.plan_version != self.ingress_plan.plan_version
        ):
            raise ValueError("Worker and ingress plans must have the same version")

        return self
