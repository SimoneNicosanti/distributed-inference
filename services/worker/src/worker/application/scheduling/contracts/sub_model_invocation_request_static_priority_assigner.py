from abc import abstractmethod
from typing import override

from scheduling.static_priority_assigner import StaticPriorityAssigner
from worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class SubModelInvocationRequestStaticPriorityAssigner(
    StaticPriorityAssigner[SubModelInvocationRequest]
):
    @abstractmethod
    @override
    def assign_priority(self, request: SubModelInvocationRequest) -> int: ...
