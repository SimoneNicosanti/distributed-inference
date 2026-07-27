from collections.abc import AsyncIterator

import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from distributed_inference.adapters.outbound.directory.redis.redis_registry import (
    RedisServerRegistry,
    RedisServiceRegistry,
)
from distributed_inference.adapters.outbound.directory.redis.redis_resolver import (
    RedisServiceResolver,
)
from test.contracts.directory_service_contract import (
    DirectoryDependencies,
    DirectoryServiceContract,
)


class TestRedisDirectoryContract(DirectoryServiceContract):
    @pytest_asyncio.fixture
    async def directory(self) -> AsyncIterator[DirectoryDependencies]:
        redis = FakeRedis()
        try:
            yield DirectoryDependencies(
                server_registry=RedisServerRegistry(redis),
                service_registry=RedisServiceRegistry(redis),
                service_resolver=RedisServiceResolver(redis),
            )
        finally:
            await redis.aclose()
