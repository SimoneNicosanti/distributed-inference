from typing import override

from directory.api.directory import ServiceDirectory
from directory.contracts.service_instance import (
    ServiceInstance,
)
from redis import asyncio as redis_asyncio
from shared.identifiers.identifiers import ServiceId


class RedisDirectory(ServiceDirectory):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def put_service_instance(self, service_instance: ServiceInstance) -> None:
        base_key = self.__build_base_key()
        service_key = self.__build_service_key(service_instance.service_id)

        await self._redis.hset(
            base_key, service_key, service_instance.model_dump_json()
        )

    @override
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance:
        base_key = self.__build_base_key()
        service_key = self.__build_service_key(service_id)

        value = await self._redis.hget(base_key, service_key)
        if value is None:
            raise KeyError(f"Service {service_id} not found")

        return ServiceInstance.model_validate_json(value)

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        base_key = self.__build_base_key()
        values = await self._redis.hvals(base_key)

        return [ServiceInstance.model_validate_json(value) for value in values]

    @staticmethod
    def __build_base_key() -> str:
        return "discovery:services"

    @staticmethod
    def __build_service_key(service_id: ServiceId) -> str:
        return str(service_id.service_id)
