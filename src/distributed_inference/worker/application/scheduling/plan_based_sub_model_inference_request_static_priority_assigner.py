from typing import override

from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.scheduling.contracts.sub_model_inference_request_static_priority_assigner import (
    SubModelInferenceRequestStaticPriorityAssigner,
)
from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceRequest,
)


class PlanBasedSubModelInferenceRequestStaticPriorityAssigner(
    SubModelInferenceRequestStaticPriorityAssigner
):
    def __init__(self, plan: ServiceInferencePlan) -> None:
        self._plan = plan

    @override
    def assign_priority(self, request: SubModelInferenceRequest) -> int:
        ## TODO: Implement this

        raise NotImplementedError
