from abc import abstractmethod

from typing_extensions import override

from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
)
from distributed_inference.building_blocks.scheduling.static_priority_assigner import (
    StaticPriorityAssigner,
)


class ActivityRequestStaticPriorityAssigner(StaticPriorityAssigner):
    @abstractmethod
    @override
    def assign_priority(self, request: ActivityRequest) -> int: ...
