from typing import override

from catalog.api.service.service_directory import ServiceDirectory
from catalog.redis.service.keys_factory import (
    build_catalog_key,
    build_service_hset_key,
)
from integration.service_instance.service_instance import ServiceInstance
from redis import asyncio as redis_asyncio
from shared.service.service import ServiceId


class RedisServiceDirectory(ServiceDirectory):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance:
        value = await self._redis.hget(
            build_catalog_key(), build_service_hset_key(service_id)
        )
        if value is None:
            raise KeyError(f"Service {service_id} not found")
        return ServiceInstance.model_validate_json(value)

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        values = await self._redis.hvals(build_catalog_key())
        return [ServiceInstance.model_validate_json(value) for value in values]
