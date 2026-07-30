from typing import override

from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.scheduling.contracts.inference_request_static_priority_assigner import (
    InferenceRequestStaticPriorityAssigner,
)
from distributed_inference.worker.domain.inference_flow import (
    InferenceRequest,
)


class PlanBasedInferenceRequestStaticPriorityAssigner(
    InferenceRequestStaticPriorityAssigner
):
    def __init__(self, plan: ServiceInferencePlan) -> None:
        self._plan = plan

    @override
    def assign_priority(self, request: InferenceRequest) -> int:
        ## TODO: Implement this

        raise NotImplementedError
