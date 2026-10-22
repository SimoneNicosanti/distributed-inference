from typing import override

from redis import asyncio as redis_asyncio

from integration.redis.worker_profiles.network_profile_keys import (
    build_network_profile_key_per_worker,
)
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.network.publisher.network_profile_mapper import (
    NetworkProfileMapper,
)
from worker.application.ports.outbound.profiling.network.network_profile_publisher import (
    NetworkProfilePublisher,
)
from worker.domain.profiling.network_profile import NetworkProfile


class RedisNetworkProfilePublisher(NetworkProfilePublisher):
    def __init__(self, worker_id: WorkerId, redis: redis_asyncio.Redis) -> None:
        self._worker_id = worker_id
        self._redis = redis

    @override
    async def publish_network_profile(self, network_profile: NetworkProfile) -> None:

        redis_key = build_network_profile_key_per_worker(self._worker_id)
        dto = NetworkProfileMapper.network_profile_to_dto(
            self._worker_id,
            network_profile,
        )
        redis_value = dto.model_dump_json()

        success = await self._redis.set(redis_key, redis_value)
        if not success:
            raise Exception("Failed to publish network profile")
