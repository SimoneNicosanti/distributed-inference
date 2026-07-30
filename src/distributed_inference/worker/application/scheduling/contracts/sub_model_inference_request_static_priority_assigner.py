from abc import abstractmethod

from typing_extensions import override

from distributed_inference.building_blocks.scheduling.static_priority_assigner import (
    StaticPriorityAssigner,
)
from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceRequest,
)


class SubModelInferenceRequestStaticPriorityAssigner(StaticPriorityAssigner):
    @abstractmethod
    @override
    def assign_priority(self, request: SubModelInferenceRequest) -> int: ...
