from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis
from pydantic import ValidationError

from distributed_inference.adapters.outbound.directory.redis.redis_registry import (
    RedisServerRegistry,
    RedisServiceRegistry,
)
from distributed_inference.adapters.outbound.directory.redis.redis_resolver import (
    RedisServiceResolver,
)
from distributed_inference.application.directory.domain.server_registration import (
    ServerRegistration,
)
from distributed_inference.application.directory.domain.service_registration import (
    ServiceRegistration,
)
from distributed_inference.domain.identifiers import ServerId
from distributed_inference.domain.service_instance import ServiceInstance
from test.contracts.directory_service_contract import (
    build_server_id,
    build_service_instance,
)


@pytest_asyncio.fixture
async def redis() -> AsyncIterator[FakeRedis]:
    client = FakeRedis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_server_registry_stores_id_under_expected_key(
    redis: FakeRedis,
) -> None:
    server_id = build_server_id()

    await RedisServerRegistry(redis).register_server(
        ServerRegistration(server_id=server_id)
    )

    stored = await redis.get(f"discovery:servers:{server_id.server_id}")

    assert stored is not None
    assert ServerId.model_validate_json(stored) == server_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_registry_stores_full_instance_under_expected_key(
    redis: FakeRedis,
) -> None:
    service = build_service_instance(build_server_id())
    service_id = service.service_id

    await RedisServiceRegistry(redis).register_service(
        ServiceRegistration(service_instance=service)
    )

    stored = await redis.get(
        f"discovery:services:{service_id.server_id.server_id}:{service_id.service_id}"
    )

    assert stored is not None
    assert ServiceInstance.model_validate_json(stored) == service


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolver_rejects_corrupt_service_payload(
    redis: FakeRedis,
) -> None:
    service = build_service_instance(build_server_id())
    service_id = service.service_id
    key = f"discovery:services:{service_id.server_id.server_id}:{service_id.service_id}"
    await redis.set(key, b'{"invalid": "service"}')

    with pytest.raises(ValidationError):
        await RedisServiceResolver(redis).resolve_service_by_id(service_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_by_server_deduplicates_scan_results(
    redis: FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_id = build_server_id()
    service = build_service_instance(server_id)
    await RedisServerRegistry(redis).register_server(
        ServerRegistration(server_id=server_id)
    )
    await RedisServiceRegistry(redis).register_service(
        ServiceRegistration(service_instance=service)
    )
    original_scan_iter = redis.scan_iter

    async def duplicate_scan_iter(*, match: str) -> AsyncIterator[bytes]:
        keys = [key async for key in original_scan_iter(match=match)]
        for key in (*keys, *keys):
            yield key

    monkeypatch.setattr(redis, "scan_iter", duplicate_scan_iter)

    resolved = await RedisServiceResolver(redis).resolve_service_by_server(server_id)

    assert resolved == [service]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_by_server_skips_service_deleted_after_scan(
    redis: FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_id = build_server_id()
    remaining_service = build_service_instance(server_id, port=8001)
    deleted_service = build_service_instance(server_id, port=8002)
    await RedisServerRegistry(redis).register_server(
        ServerRegistration(server_id=server_id)
    )
    for service in (remaining_service, deleted_service):
        await RedisServiceRegistry(redis).register_service(
            ServiceRegistration(service_instance=service)
        )

    remaining_key = (
        "discovery:services:"
        f"{server_id.server_id}:"
        f"{remaining_service.service_id.service_id}"
    )
    remaining_payload = await redis.get(remaining_key)
    assert remaining_payload is not None

    async def mget_with_deleted_service(
        _keys: object,
    ) -> list[bytes | str | None]:
        return [remaining_payload, None]

    monkeypatch.setattr(redis, "mget", mget_with_deleted_service)

    resolved = await RedisServiceResolver(redis).resolve_service_by_server(server_id)

    assert resolved == [remaining_service]
