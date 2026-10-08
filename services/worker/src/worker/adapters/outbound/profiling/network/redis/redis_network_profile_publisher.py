from typing import override

from integration.worker.profiling.redis.network_profile_keys import (
    build_network_profile_key_per_worker,
)
from redis import asyncio as redis_asyncio
from shared.service.service import WorkerId
from worker.application.ports.outbound.profiling.network.network_profile_publisher import (
    NetworkProfilePublisher,
)
from worker.domain.profiling.network_profile import NetworkProfile


class RedisNetworkProfilePublisher(NetworkProfilePublisher):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def publish_network_profile(
        self, worker_id: WorkerId, network_profile: NetworkProfile
    ) -> None:

        redis_key = build_network_profile_key_per_worker(worker_id)
        redis_value = network_profile.model_dump_json()

        success = await self._redis.set(redis_key, redis_value)
        if not success:
            raise Exception("Failed to publish network profile")
