import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from controller.adapters.outbound.plan_version.redis_plan_version_factory import (
    RedisPlanVersionFactory,
)
from controller.composition.default_composition_root import DefaultCompositionRoot
from shared.plan.plan_version import PlanVersion


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[FakeRedis]:
    async with FakeRedis() as client:
        yield client


@pytest.mark.asyncio
async def test_allocates_versions_from_empty_store(redis: FakeRedis) -> None:
    factory = RedisPlanVersionFactory(redis)

    assert await factory.build_next_plan_version() == PlanVersion(1)
    assert await factory.build_next_plan_version() == PlanVersion(2)


@pytest.mark.asyncio
async def test_concurrent_clients_allocate_unique_versions() -> None:
    server = FakeServer()
    async with FakeRedis(server=server) as first, FakeRedis(server=server) as second:
        factories = (RedisPlanVersionFactory(first), RedisPlanVersionFactory(second))
        versions = await asyncio.wait_for(
            asyncio.gather(
                *(factories[index % 2].build_next_plan_version() for index in range(64))
            ),
            timeout=5,
        )

    assert sorted(version.version_number for version in versions) == list(range(1, 65))


@pytest.mark.asyncio
async def test_recreated_factory_resumes_from_persisted_version() -> None:
    server = FakeServer()
    async with FakeRedis(server=server) as first:
        await first.set("current-plan-version", 41)
        assert await RedisPlanVersionFactory(
            first
        ).build_next_plan_version() == PlanVersion(42)

    async with FakeRedis(server=server) as second:
        assert await RedisPlanVersionFactory(
            second
        ).build_next_plan_version() == PlanVersion(43)


@pytest.mark.asyncio
async def test_composition_uses_redis_factory_and_manages_actuation_lifecycle(
    redis: FakeRedis,
) -> None:
    with patch(
        "controller.composition.default_composition_root.DefaultActuationManager",
        autospec=True,
    ) as manager_type:
        manager = manager_type.return_value
        async with DefaultCompositionRoot().build_actuation_manager_context(
            AsyncMock(), AsyncMock(), AsyncMock(), redis
        ):
            factory = manager_type.call_args.args[2]
            assert isinstance(factory, RedisPlanVersionFactory)
            assert await factory.build_next_plan_version() == PlanVersion(1)
            assert await RedisPlanVersionFactory(
                redis
            ).build_next_plan_version() == PlanVersion(2)
            manager.start.assert_awaited_once()
            manager.stop.assert_not_awaited()

        manager.stop.assert_awaited_once()
