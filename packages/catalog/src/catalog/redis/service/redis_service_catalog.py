from typing import override

from catalog.api.service.service_directory import ServiceDirectory
from catalog.api.service.service_registrar import ServiceRegistrar
from catalog.redis.service.keys_factory import (
    build_catalog_key,
    build_service_hset_key,
)
from integration.service_instance.service_instance import (
    ServiceInstance,
)
from redis import asyncio as redis_asyncio
from shared.service.service import ServiceId


class RedisServiceCatalog(ServiceDirectory, ServiceRegistrar):
    def __init__(self, redis: redis_asyncio.Redis, entry_ttl_s: int) -> None:
        self._redis = redis
        if entry_ttl_s <= 0:
            raise ValueError("entry_ttl must be positive")
        self._entry_ttl = entry_ttl_s

    @override
    async def register_or_renew(self, service_instance: ServiceInstance) -> None:
        catalog_key = build_catalog_key()
        service_hset_key = build_service_hset_key(service_instance.service_id)

        pipeline = self._redis.pipeline(transaction=True)
        pipeline.hset(
            catalog_key,
            service_hset_key,
            service_instance.model_dump_json(),
        )
        pipeline.hexpire(
            catalog_key,
            self._entry_ttl,
            service_hset_key,
        )
        await pipeline.execute()

    @override
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance:
        catalog_key = build_catalog_key()
        service_hset_key = build_service_hset_key(service_id)

        value = await self._redis.hget(catalog_key, service_hset_key)
        if value is None:
            raise KeyError(f"Service {service_id} not found")

        return ServiceInstance.model_validate_json(value)

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        catalog_key = build_catalog_key()
        values = await self._redis.hvals(catalog_key)

        return [ServiceInstance.model_validate_json(value) for value in values]
