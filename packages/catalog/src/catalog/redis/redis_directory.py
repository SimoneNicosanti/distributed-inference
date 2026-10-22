from typing import override

from catalog.api.directory import Directory
from catalog.redis.keys_factory import (
    build_catalog_key,
    build_service_hset_key,
)
from integration.service_instance.service_instance import (
    ServiceInstance,
)
from redis import asyncio as redis_asyncio
from shared.service.service import ServiceId


class RedisDirectory(Directory):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

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
