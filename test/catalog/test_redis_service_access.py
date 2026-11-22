from unittest.mock import AsyncMock, MagicMock

import pytest
from redis import asyncio as redis_asyncio

from catalog.api.service.service_directory import ServiceInstanceReadError
from catalog.redis.service.redis_service_directory import RedisServiceDirectory
from catalog.redis.service.redis_service_registrar import RedisServiceRegistrar
from integration.service_instance.service_instance import ServiceInstance
from redis_access.address_factory.service.service_instance_address_factory import (
    HashMapServiceInstanceAddressFactory,
)
from redis_access.redis_reader.hash_map_redis_reader import HashMapRedisReader
from shared.service.service import ServiceId


def _directory(redis: redis_asyncio.Redis) -> RedisServiceDirectory:
    return RedisServiceDirectory(
        redis, HashMapRedisReader(), HashMapServiceInstanceAddressFactory()
    )


@pytest.fixture
def service_instance() -> ServiceInstance:
    return ServiceInstance(service_id=ServiceId(id="worker-a"), capabilities=[])


@pytest.mark.asyncio
async def test_directory_reads_without_ttl(service_instance: ServiceInstance) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hget = AsyncMock(return_value=service_instance.model_dump_json())
    directory = _directory(redis)

    result = await directory.get_service_instance_by_service_id(
        service_instance.service_id
    )

    assert result == service_instance
    redis.hget.assert_awaited_once_with(
        "catalog:services", str(service_instance.service_id.id)
    )
    redis.pipeline.assert_not_called()
    assert not hasattr(directory, "register_or_renew")


@pytest.mark.asyncio
async def test_directory_rejects_missing_service() -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hget = AsyncMock(return_value=None)

    with pytest.raises(ServiceInstanceReadError):
        await _directory(redis).get_service_instance_by_service_id(
            ServiceId(id="missing")
        )


@pytest.mark.asyncio
async def test_directory_lists_services(service_instance: ServiceInstance) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hvals = AsyncMock(return_value=[service_instance.model_dump_json()])

    assert await _directory(redis).get_all_service_instances() == [service_instance]
    redis.hvals.assert_awaited_once_with("catalog:services")


@pytest.mark.asyncio
async def test_directory_lists_empty_catalog() -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hvals = AsyncMock(return_value=[])

    assert await _directory(redis).get_all_service_instances() == []


@pytest.mark.parametrize("ttl", [0, -1])
def test_registrar_requires_positive_ttl(ttl: int) -> None:
    with pytest.raises(ValueError, match="entry_ttl must be positive"):
        RedisServiceRegistrar(
            AsyncMock(spec=redis_asyncio.Redis),
            entry_ttl_s=ttl,
            address_factory=HashMapServiceInstanceAddressFactory(),
        )


@pytest.mark.asyncio
async def test_registrar_writes_and_renews_field_atomically(
    service_instance: ServiceInstance,
) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    pipeline = MagicMock()
    pipeline.execute = AsyncMock()
    redis.pipeline.return_value = pipeline
    registrar = RedisServiceRegistrar(
        redis,
        entry_ttl_s=60,
        address_factory=HashMapServiceInstanceAddressFactory(),
    )

    await registrar.register_or_renew(service_instance)

    redis.pipeline.assert_called_once_with(transaction=True)
    pipeline.hset.assert_called_once_with(
        "catalog:services",
        str(service_instance.service_id.id),
        service_instance.model_dump_json(),
    )
    pipeline.hexpire.assert_called_once_with(
        "catalog:services", 60, str(service_instance.service_id.id)
    )
    pipeline.execute.assert_awaited_once_with()
    assert not hasattr(registrar, "get_all_service_instances")
