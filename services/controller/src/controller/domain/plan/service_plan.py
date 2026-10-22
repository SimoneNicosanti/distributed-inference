from dataclasses import dataclass

from controller.domain.plan.ingress_plan import IngressPlan
from controller.domain.plan.worker_plan import WorkerPlan


@dataclass(frozen=True, slots=True)
class ServicePlan:
    worker_plan: WorkerPlan
    ingress_plan: IngressPlan | None

    def __post_init__(self) -> None:
        if (
            self.ingress_plan is not None
            and self.worker_plan.plan_version != self.ingress_plan.plan_version
        ):
            raise ValueError("Worker and ingress plans must have the same version")
