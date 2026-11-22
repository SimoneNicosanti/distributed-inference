from typing import override

from redis import RedisError
from redis import asyncio as redis_asyncio

from redis_access.address_factory.worker_profiles.resource_profile_address_factory import (
    ResourceProfileAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.publisher.mappers.resource_profile_mapper import (
    ResourceProfileMapper,
)
from worker.application.ports.outbound.profiling.resource.resource_profile_publisher import (
    ResourceProfilePublisher,
    ResourceProfileWriteError,
)
from worker.domain.profiling.resource_profile import ResourceProfile


class RedisResourceProfilePublisher(ResourceProfilePublisher):
    def __init__(
        self,
        worker_id: WorkerId,
        redis: redis_asyncio.Redis,
        resource_profile_address_factory_cls: type[
            ResourceProfileAddressFactory[HashMapRedisAddress]
        ],
    ) -> None:
        self._worker_id = worker_id
        self._redis = redis
        self._resource_profile_address_factory_cls = (
            resource_profile_address_factory_cls
        )

    @override
    async def publish_resource_profile(self, resource_profile: ResourceProfile) -> None:
        try:
            addr = self._resource_profile_address_factory_cls.build_worker_resource_profile_address(
                self._worker_id
            )
            redis_value = ResourceProfileMapper.resource_profile_to_dto(
                self._worker_id,
                resource_profile,
            ).model_dump_json()
            await self._redis.hset(addr.hash_map_key, addr.field_key, redis_value)

        except (ValueError, RedisError) as e:
            raise ResourceProfileWriteError("Unable to publish resource profile") from e
