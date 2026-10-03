from typing import override

from redis import asyncio as redis_asyncio

from shared.service.service import WorkerId
from worker.application.ports.outbound.profiling.resource.resource_profile_publisher import (
    ResourceProfilePublisher,
)
from worker.domain.profiling.resource_profile import ResourceProfile


class RedisResourceProfilePublisher(ResourceProfilePublisher):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def publish_resource_profile(
        self, worker_id: WorkerId, resource_profile: ResourceProfile
    ) -> None:

        redis_key = self.__build_redis_key(worker_id)
        redis_value = resource_profile.model_dump_json()

        success = await self._redis.set(redis_key, redis_value)
        if not success:
            raise Exception("Failed to publish resource profile")

    def __build_redis_key(self, worker_id: WorkerId) -> str:
        return f"worker:{worker_id.service_id}:resource-profile"
