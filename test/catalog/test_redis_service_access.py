from unittest.mock import AsyncMock, MagicMock

import pytest
from redis import asyncio as redis_asyncio

from catalog.redis.service.keys_factory import build_catalog_key, build_service_hset_key
from catalog.redis.service.redis_service_directory import RedisServiceDirectory
from catalog.redis.service.redis_service_registrar import RedisServiceRegistrar
from integration.service_instance.service_instance import ServiceInstance
from shared.service.service import ServiceId


@pytest.fixture
def service_instance() -> ServiceInstance:
    return ServiceInstance(service_id=ServiceId(id="worker-a"), capabilities=[])


@pytest.mark.asyncio
async def test_directory_reads_without_ttl(service_instance: ServiceInstance) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hget.return_value = service_instance.model_dump_json()
    directory = RedisServiceDirectory(redis)

    result = await directory.get_service_instance_by_service_id(
        service_instance.service_id
    )

    assert result == service_instance
    redis.hget.assert_awaited_once_with(
        build_catalog_key(), build_service_hset_key(service_instance.service_id)
    )
    redis.pipeline.assert_not_called()
    assert not hasattr(directory, "register_or_renew")


@pytest.mark.asyncio
async def test_directory_rejects_missing_service() -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hget.return_value = None

    with pytest.raises(KeyError, match="not found"):
        await RedisServiceDirectory(redis).get_service_instance_by_service_id(
            ServiceId(id="missing")
        )


@pytest.mark.asyncio
async def test_directory_lists_services(service_instance: ServiceInstance) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hvals.return_value = [service_instance.model_dump_json()]

    assert await RedisServiceDirectory(redis).get_all_service_instances() == [
        service_instance
    ]
    redis.hvals.assert_awaited_once_with(build_catalog_key())


@pytest.mark.asyncio
async def test_directory_lists_empty_catalog() -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.hvals.return_value = []

    assert await RedisServiceDirectory(redis).get_all_service_instances() == []


@pytest.mark.parametrize("ttl", [0, -1])
def test_registrar_requires_positive_ttl(ttl: int) -> None:
    with pytest.raises(ValueError, match="entry_ttl must be positive"):
        RedisServiceRegistrar(AsyncMock(spec=redis_asyncio.Redis), entry_ttl_s=ttl)


@pytest.mark.asyncio
async def test_registrar_writes_and_renews_field_atomically(
    service_instance: ServiceInstance,
) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    pipeline = MagicMock()
    pipeline.execute = AsyncMock()
    redis.pipeline.return_value = pipeline
    registrar = RedisServiceRegistrar(redis, entry_ttl_s=60)

    await registrar.register_or_renew(service_instance)

    redis.pipeline.assert_called_once_with(transaction=True)
    pipeline.hset.assert_called_once_with(
        build_catalog_key(),
        build_service_hset_key(service_instance.service_id),
        service_instance.model_dump_json(),
    )
    pipeline.hexpire.assert_called_once_with(
        build_catalog_key(), 60, build_service_hset_key(service_instance.service_id)
    )
    pipeline.execute.assert_awaited_once_with()
    assert not hasattr(registrar, "get_all_service_instances")
