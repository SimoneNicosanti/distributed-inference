import asyncio
from unittest.mock import MagicMock, patch

import pytest

from distributed_inference.activity_manager.application.scheduling.contracts.activity_request_static_priority_assigner import (
    ActivityRequestStaticPriorityAssigner,
)
from distributed_inference.activity_manager.application.scheduling.default_activity_scheduler import (
    DefaultActivityRequestScheduler,
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
@pytest.mark.asyncio
async def test_scheduler_uses_assigner_contract_and_dequeues_lowest_priority() -> None:
    assigner = MagicMock(spec=ActivityRequestStaticPriorityAssigner)
    assigner.assign_priority.side_effect = [10, 0]
    scheduler = DefaultActivityRequestScheduler(assigner)
    loop = asyncio.get_running_loop()
    low_priority_request = _request(ActivityType.PROFILING_NETWORK)
    high_priority_request = _request(ActivityType.INFERENCE_EXECUTION)
    low_priority_future = loop.create_future()
    high_priority_future = loop.create_future()

    await scheduler.enqueue(low_priority_request, low_priority_future)
    await scheduler.enqueue(high_priority_request, high_priority_future)
    assert await scheduler.length() == 2

    first_request, first_future = await scheduler.dequeue()
    second_request, second_future = await scheduler.dequeue()

    assert (first_request, first_future) == (
        high_priority_request,
        high_priority_future,
    )
    assert (second_request, second_future) == (
        low_priority_request,
        low_priority_future,
    )
    assert scheduler._priority_queue.empty()
    assert await scheduler.length() == 0
    assert assigner.assign_priority.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scheduler_preserves_enqueue_order_for_complete_ties() -> None:
    assigner = MagicMock(spec=ActivityRequestStaticPriorityAssigner)
    assigner.assign_priority.return_value = 0
    scheduler = DefaultActivityRequestScheduler(assigner)
    loop = asyncio.get_running_loop()
    first_request = _request(ActivityType.INFERENCE_EXECUTION)
    second_request = _request(ActivityType.INFERENCE_FORWARDING)
    first_future = loop.create_future()
    second_future = loop.create_future()

    with patch(
        "distributed_inference.activity_manager.application.scheduling."
        "default_activity_scheduler.time.monotonic_ns",
        return_value=1_000,
    ):
        await scheduler.enqueue(first_request, first_future)
        await scheduler.enqueue(second_request, second_future)

    assert await scheduler.dequeue() == (first_request, first_future)
    assert await scheduler.dequeue() == (second_request, second_future)
