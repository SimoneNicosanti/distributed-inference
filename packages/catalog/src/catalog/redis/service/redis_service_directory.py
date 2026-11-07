import logging
from typing import override

from pydantic import ValidationError

from catalog.api.service.service_directory import (
    ServiceDirectory,
    ServiceInstanceReadError,
)
from catalog.redis.service.keys_factory import (
    build_catalog_key,
    build_service_hset_key,
)
from integration.service_instance.service_instance import ServiceInstance
from redis import RedisError
from redis import asyncio as redis_asyncio
from shared.service.service import ServiceId

logger = logging.getLogger(__name__)


class RedisServiceDirectory(ServiceDirectory):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance:
        try:
            value = await self._redis.hget(
                build_catalog_key(), build_service_hset_key(service_id)
            )
            if value is None:
                raise KeyError(f"Service {service_id} not found")
            instance = ServiceInstance.model_validate_json(value)

            if instance.service_id != service_id:
                raise ValueError("Service instance ID mismatch")
            return instance
        except (KeyError, ValueError, RedisError) as e:
            raise ServiceInstanceReadError(
                f"Could not read service instance for {service_id.id}"
            ) from e

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        try:
            values = await self._redis.hvals(build_catalog_key())

            return self._build_instances_list(values)
        except (ValueError, RedisError) as e:
            raise ServiceInstanceReadError(
                "Could not read service instance list"
            ) from e

    def _build_instances_list(self, values: list[bytes | str]) -> list[ServiceInstance]:
        instances: list[ServiceInstance] = []
        for value in values:
            try:
                instance = ServiceInstance.model_validate_json(value)
                instances.append(instance)
            except ValidationError:
                logger.warning(
                    "Ignoring invalid service instance in catalog",
                    exc_info=True,
                )

        return instances
