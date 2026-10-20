from typing import override

from integration.worker.profiling.redis.network_profile_dto import (
    NetworkConnectionDto,
    NetworkProfileDto,
)
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
    def __init__(self, worker_id: WorkerId, redis: redis_asyncio.Redis) -> None:
        self._worker_id = worker_id
        self._redis = redis

    @override
    async def publish_network_profile(self, network_profile: NetworkProfile) -> None:

        redis_key = build_network_profile_key_per_worker(self._worker_id)
        dto = NetworkProfileDto(
            source_worker_id=self._worker_id,
            connections=tuple(
                NetworkConnectionDto(
                    target_worker_id=connection.worker_id,
                    throughput_bytes_per_sec=(
                        connection.connection_info.throughput_bytes_per_sec
                    ),
                    round_trip_time_s=connection.connection_info.round_trip_time_s,
                )
                for connection in network_profile.connections
            ),
        )
        redis_value = dto.model_dump_json()

        success = await self._redis.set(redis_key, redis_value)
        if not success:
            raise Exception("Failed to publish network profile")
