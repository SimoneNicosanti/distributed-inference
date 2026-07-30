import time
from typing import override

from distributed_inference.activity_manager.application.scheduling.contracts.activity_request_static_priority_assigner import (
    ActivityRequestStaticPriorityAssigner,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
    ActivityType,
)


class DefaultActivityRequestStaticPriorityAssigner(
    ActivityRequestStaticPriorityAssigner
):
    def __init__(self, time_slot_length: int):
        if time_slot_length <= 0:
            raise ValueError("Time slot length must be greater than 0")
        self._time_slot_length = time_slot_length
        self._initial_time = time.monotonic_ns()

    @override
    def assign_priority(self, request: ActivityRequest) -> int:
        match request.activity_type:
            case ActivityType.INFERENCE_EXECUTION:
                base_priority = 0
            case ActivityType.INFERENCE_FORWARDING:
                base_priority = 0
            case ActivityType.PROFILING_EXECUTION:
                base_priority = 5
            case ActivityType.PROFILING_NETWORK:
                base_priority = 10

        current_slot = (
            time.monotonic_ns() - self._initial_time
        ) // self._time_slot_length

        priority = base_priority + current_slot

        return priority
