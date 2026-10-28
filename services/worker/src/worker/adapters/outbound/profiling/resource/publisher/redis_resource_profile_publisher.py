from typing import override

from redis import asyncio as redis_asyncio

from integration.redis.worker_profiles.resource_profile_keys import (
    ResourceProfileKeyFactory,
)
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.resource.publisher.resource_profile_mapper import (
    ResourceProfileMapper,
)
from worker.application.ports.outbound.profiling.resource.resource_profile_publisher import (
    ResourceProfilePublisher,
)
from worker.domain.profiling.resource_profile import ResourceProfile


class RedisResourceProfilePublisher(ResourceProfilePublisher):
    def __init__(
        self,
        worker_id: WorkerId,
        redis: redis_asyncio.Redis,
        resource_profile_key_factory_cls: type[ResourceProfileKeyFactory],
    ) -> None:
        self._worker_id = worker_id
        self._redis = redis
        self._resource_profile_key_factory_cls = resource_profile_key_factory_cls

    @override
    async def publish_resource_profile(self, resource_profile: ResourceProfile) -> None:
        redis_key = await self._resource_profile_key_factory_cls.build_resource_profile_key_per_worker(
            self._worker_id
        )
        redis_value = ResourceProfileMapper.resource_profile_to_dto(
            self._worker_id,
            resource_profile,
        ).model_dump_json()

        success = await self._redis.set(redis_key, redis_value)
        if not success:
            raise Exception("Failed to publish resource profile")
