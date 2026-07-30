from unittest.mock import patch

import pytest

from distributed_inference.activity_manager.application.scheduling.default_activity_request_static_priority_assigner import (
    DefaultActivityRequestStaticPriorityAssigner,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
    ActivityType,
)


def _request(activity_type: ActivityType) -> ActivityRequest:
    return ActivityRequest(
        activity_type=activity_type,
        activity_resources={},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("activity_type", "expected_priority"),
    [
        (ActivityType.INFERENCE_EXECUTION, 0),
        (ActivityType.INFERENCE_FORWARDING, 0),
        (ActivityType.PROFILING_EXECUTION, 5),
        (ActivityType.PROFILING_NETWORK, 10),
    ],
)
def test_assign_priority_uses_activity_type_base_priority(
    activity_type: ActivityType,
    expected_priority: int,
) -> None:
    with patch(
        "distributed_inference.activity_manager.application.scheduling."
        "default_activity_request_static_priority_assigner.time.monotonic_ns",
        side_effect=[1_000, 1_000],
    ):
        assigner = DefaultActivityRequestStaticPriorityAssigner(
            time_slot_length=100,
        )

        assert assigner.assign_priority(_request(activity_type)) == expected_priority


@pytest.mark.unit
def test_assign_priority_adds_elapsed_time_slots() -> None:
    with patch(
        "distributed_inference.activity_manager.application.scheduling."
        "default_activity_request_static_priority_assigner.time.monotonic_ns",
        side_effect=[1_000, 1_350],
    ):
        assigner = DefaultActivityRequestStaticPriorityAssigner(
            time_slot_length=100,
        )

        assert assigner.assign_priority(_request(ActivityType.PROFILING_EXECUTION)) == 8


@pytest.mark.unit
@pytest.mark.parametrize("time_slot_length", [0, -1])
def test_constructor_rejects_non_positive_time_slot_length(
    time_slot_length: int,
) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        DefaultActivityRequestStaticPriorityAssigner(time_slot_length)
