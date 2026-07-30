from abc import abstractmethod

from typing_extensions import override

from distributed_inference.building_blocks.scheduling.static_priority_assigner import (
    StaticPriorityAssigner,
)
from distributed_inference.worker.domain.inference_flow import (
    InferenceRequest,
)


class InferenceRequestStaticPriorityAssigner(StaticPriorityAssigner):
    @abstractmethod
    @override
    def assign_priority(self, request: InferenceRequest) -> int: ...
