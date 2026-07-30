import asyncio
from unittest.mock import MagicMock

import pytest

from distributed_inference.activity_manager.application.ports.outbound.resource_manager import (
    ResourceManager,
)
from distributed_inference.activity_manager.application.scheduling.contracts.activity_request_scheduler import (
    ActivityRequestScheduler,
)
from distributed_inference.activity_manager.application.services.default_activity_manager import (
    DefaultActivityManager,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityGrantId,
    ActivityGrantInfo,
    ActivityRequest,
    ActivityType,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceLease,
    ResourceLeaseId,
    ResourceRequirement,
    ResourceType,
)


def _request() -> ActivityRequest:
    return ActivityRequest(
        activity_type=ActivityType.INFERENCE_EXECUTION,
        activity_resources={
            ResourceType.COMPUTE: ResourceRequirement(
                quantity=1,
                exclusive=False,
            )
        },
    )


def _manager() -> tuple[
    DefaultActivityManager,
    MagicMock,
    MagicMock,
]:
    scheduler = MagicMock(spec=ActivityRequestScheduler)
    resource_manager = MagicMock(spec=ResourceManager)
    return (
        DefaultActivityManager(scheduler, resource_manager),
        scheduler,
        resource_manager,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_activity_grant_info_enqueues_and_awaits_result() -> None:
    manager, scheduler, _ = _manager()
    request = _request()
    expected = ActivityGrantInfo(activity_grant_id=ActivityGrantId())

    async def complete_request(
        queued_request: ActivityRequest,
        future: asyncio.Future[ActivityGrantInfo],
    ) -> None:
        assert queued_request == request
        future.set_result(expected)

    scheduler.enqueue.side_effect = complete_request

    result = await manager.get_activity_grant_info(request)

    assert result == expected
    scheduler.enqueue.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_activity_grant_releases_known_lease_once() -> None:
    manager, _, resource_manager = _manager()
    grant_id = ActivityGrantId()
    lease = ResourceLease(resource_lease_id=ResourceLeaseId())
    manager._pending_activity_grants[grant_id] = lease

    await manager.release_activity_grant(grant_id)
    await manager.release_activity_grant(grant_id)

    resource_manager.release_resource_lease.assert_awaited_once_with(
        lease.resource_lease_id
    )
    assert grant_id not in manager._pending_activity_grants


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_acquires_resources_and_completes_grant_future() -> None:
    manager, scheduler, resource_manager = _manager()
    request = _request()
    grant_future: asyncio.Future[ActivityGrantInfo] = (
        asyncio.get_running_loop().create_future()
    )
    lease = ResourceLease(resource_lease_id=ResourceLeaseId())
    scheduler.dequeue.side_effect = [
        (request, grant_future),
        asyncio.CancelledError(),
    ]
    resource_manager.acquire_resource_lease.return_value = lease

    with pytest.raises(asyncio.CancelledError):
        await manager.start()

    grant_info = grant_future.result()
    resource_manager.acquire_resource_lease.assert_awaited_once_with(
        request.activity_resources
    )
    assert manager._pending_activity_grants[grant_info.activity_grant_id] == lease


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_failure_keeps_grant_available_for_retry() -> None:
    manager, _, resource_manager = _manager()
    grant_id = ActivityGrantId()
    lease = ResourceLease(resource_lease_id=ResourceLeaseId())
    manager._pending_activity_grants[grant_id] = lease
    resource_manager.release_resource_lease.side_effect = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        await manager.release_activity_grant(grant_id)

    assert manager._pending_activity_grants[grant_id] == lease

    resource_manager.release_resource_lease.side_effect = None
    await manager.release_activity_grant(grant_id)

    assert grant_id not in manager._pending_activity_grants
    assert resource_manager.release_resource_lease.await_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_skips_request_cancelled_before_resource_acquisition() -> None:
    manager, scheduler, resource_manager = _manager()
    request = _request()
    grant_future: asyncio.Future[ActivityGrantInfo] = (
        asyncio.get_running_loop().create_future()
    )
    grant_future.cancel()
    scheduler.dequeue.side_effect = [
        (request, grant_future),
        asyncio.CancelledError(),
    ]

    with pytest.raises(asyncio.CancelledError):
        await manager.start()

    resource_manager.acquire_resource_lease.assert_not_awaited()
    assert manager._pending_activity_grants == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_releases_lease_when_request_is_cancelled_during_acquisition() -> (
    None
):
    manager, scheduler, resource_manager = _manager()
    request = _request()
    grant_future: asyncio.Future[ActivityGrantInfo] = (
        asyncio.get_running_loop().create_future()
    )
    lease = ResourceLease(resource_lease_id=ResourceLeaseId())
    scheduler.dequeue.side_effect = [
        (request, grant_future),
        asyncio.CancelledError(),
    ]

    async def cancel_request_during_acquisition(
        _resource_lock: object,
    ) -> ResourceLease:
        grant_future.cancel()
        return lease

    resource_manager.acquire_resource_lease.side_effect = (
        cancel_request_during_acquisition
    )

    with pytest.raises(asyncio.CancelledError):
        await manager.start()

    resource_manager.release_resource_lease.assert_awaited_once_with(
        lease.resource_lease_id
    )
    assert manager._pending_activity_grants == {}
