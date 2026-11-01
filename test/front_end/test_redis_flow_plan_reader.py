from unittest.mock import AsyncMock

import pytest

from front_end.adapters.outbound.redis_flow_plan_reader import RedisFlowPlanReader
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["3", b"3"])
async def test_reads_current_version_from_string_or_bytes(value: str | bytes) -> None:
    flow_id = FlowId(user_id=UserId())
    redis = AsyncMock()
    redis.get.return_value = value
    factory = StringFlowPlanKeyFactory()
    reader = RedisFlowPlanReader(redis, factory)

    version = await reader.get_current_plan_version_for_flow(flow_id)

    assert version == PlanVersion(version_number=3)
    redis.get.assert_awaited_once_with(await factory.build_current_plan_key(flow_id))


@pytest.mark.asyncio
async def test_missing_current_version_raises_key_error() -> None:
    redis = AsyncMock()
    redis.get.return_value = None
    reader = RedisFlowPlanReader(redis, StringFlowPlanKeyFactory())

    with pytest.raises(KeyError, match="No plan for flow id"):
        await reader.get_current_plan_version_for_flow(FlowId(user_id=UserId()))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "invalid", "1.5", "0", "-1"])
async def test_invalid_current_version_raises_value_error(value: str) -> None:
    redis = AsyncMock()
    redis.get.return_value = value
    reader = RedisFlowPlanReader(redis, StringFlowPlanKeyFactory())

    with pytest.raises(ValueError):
        await reader.get_current_plan_version_for_flow(FlowId(user_id=UserId()))
