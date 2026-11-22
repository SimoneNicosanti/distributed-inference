import logging
from typing import override

from pydantic import ValidationError

from catalog.api.service.service_directory import (
    ServiceDirectory,
    ServiceInstanceReadError,
)
from integration.service_instance.service_instance import ServiceInstance
from redis import RedisError
from redis import asyncio as redis_asyncio
from redis_access.address_factory.service.service_instance_address_factory import (
    ServiceInstanceAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.redis_reader.abc.redis_reader import RedisReader, RedisReadError
from shared.service.service import ServiceId

logger = logging.getLogger(__name__)


class RedisServiceDirectory(ServiceDirectory):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        redis_reader: RedisReader[HashMapRedisAddress],
        address_factory: ServiceInstanceAddressFactory[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        self._redis_reader = redis_reader
        self._address_factory = address_factory

    @override
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance:

        try:
            addr = self._address_factory.build_service_instance_address(service_id)
            value = await self._redis_reader.read(self._redis, addr)
            instance = ServiceInstance.model_validate_json(value)

            if instance.service_id != service_id:
                raise ValueError("Service instance ID mismatch")
            return instance

        except (RedisReadError, ValueError) as e:
            raise ServiceInstanceReadError(
                f"Could not read service instance for {service_id.id}"
            ) from e

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        try:
            addr = self._address_factory.build_service_catalog_address()
            values = await self._redis.hvals(addr.key)

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
