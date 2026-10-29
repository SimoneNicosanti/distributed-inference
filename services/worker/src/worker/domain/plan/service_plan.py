from dataclasses import dataclass

from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.sink.sink_plan import SinkPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


@dataclass(frozen=True, slots=True)
class ServicePlan:
    worker_plan: WorkerPlan
    ingress_plan: IngressPlan | None
    sink_plan: SinkPlan | None = None

    def __post_init__(self) -> None:
        plan_versions = {
            plan.plan_version
            for plan in (self.worker_plan, self.ingress_plan, self.sink_plan)
            if plan is not None
        }
        if len(plan_versions) != 1:
            raise ValueError("Service sub-plans must have the same version")
