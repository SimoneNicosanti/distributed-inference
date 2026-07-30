from unittest.mock import AsyncMock

import pytest

from distributed_inference.activity_manager.domain.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityGrantInfo,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_activity_grant_release_is_idempotent() -> None:
    grant_id = ActivityGrantId()
    release_callback = AsyncMock()
    grant = ActivityGrant(
        ActivityGrantInfo(activity_grant_id=grant_id),
        release_callback,
    )

    await grant.release()
    await grant.release()

    release_callback.assert_awaited_once_with(grant_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_activity_grant_context_releases_on_exception() -> None:
    grant_id = ActivityGrantId()
    release_callback = AsyncMock()
    grant = ActivityGrant(
        ActivityGrantInfo(activity_grant_id=grant_id),
        release_callback,
    )

    with pytest.raises(RuntimeError, match="activity failed"):
        async with grant:
            raise RuntimeError("activity failed")

    release_callback.assert_awaited_once_with(grant_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_activity_grant_can_retry_failed_release() -> None:
    grant_id = ActivityGrantId()
    release_callback = AsyncMock(
        side_effect=[RuntimeError("release failed"), None],
    )
    grant = ActivityGrant(
        ActivityGrantInfo(activity_grant_id=grant_id),
        release_callback,
    )

    with pytest.raises(RuntimeError, match="release failed"):
        await grant.release()

    await grant.release()

    assert release_callback.await_count == 2
