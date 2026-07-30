from abc import abstractmethod

from typing_extensions import override

from distributed_inference.building_blocks.scheduling.static_priority_assigner import (
    StaticPriorityAssigner,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class SubModelInvocationRequestStaticPriorityAssigner(StaticPriorityAssigner):
    @abstractmethod
    @override
    def assign_priority(self, request: SubModelInvocationRequest) -> int: ...
