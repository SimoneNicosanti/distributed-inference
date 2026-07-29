from typing import override

from distributed_inference.application.worker.contracts.execution.inference.inference_request_priority_assigner import (
    InferenceRequestPriorityAssigner,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceRequest,
)
from distributed_inference.domain.plan import ServiceInferencePlan


class DefaultInferenceRequestPriorityAssigner(InferenceRequestPriorityAssigner):
    def __init__(self, plan: ServiceInferencePlan) -> None:
        self._plan = plan

    @override
    def compute_priority(self, inference_request: InferenceRequest) -> int:
        ## TODO: Implement this

        raise NotImplementedError
