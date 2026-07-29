from typing import override

from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.scheduling.contracts.inference_request_priority_assigner import (
    InferenceRequestPriorityAssigner,
)
from distributed_inference.worker.domain.inference_flow import (
    InferenceRequest,
)


class DefaultInferenceRequestPriorityAssigner(InferenceRequestPriorityAssigner):
    def __init__(self, plan: ServiceInferencePlan) -> None:
        self._plan = plan

    @override
    def compute_priority(self, inference_request: InferenceRequest) -> int:
        ## TODO: Implement this

        raise NotImplementedError
