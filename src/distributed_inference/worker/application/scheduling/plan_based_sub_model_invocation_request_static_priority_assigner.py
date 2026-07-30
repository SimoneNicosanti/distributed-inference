from typing import override

from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.scheduling.contracts.sub_model_invocation_request_static_priority_assigner import (
    SubModelInvocationRequestStaticPriorityAssigner,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class PlanBasedSubModelInvocationRequestStaticPriorityAssigner(
    SubModelInvocationRequestStaticPriorityAssigner
):
    def __init__(self, plan: ServiceInferencePlan) -> None:
        self._plan = plan

    @override
    def assign_priority(self, request: SubModelInvocationRequest) -> int:
        ## TODO: Implement this

        raise NotImplementedError
