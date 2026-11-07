from typing import override

from redis import RedisError
from redis import asyncio as redis_asyncio

from integration.redis.worker_profiles.network_profile_keys import (
    NetworkProfileKeyFactory,
)
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.publisher.mappers.network_profile_mapper import (
    NetworkProfileMapper,
)
from worker.application.ports.outbound.profiling.network.network_profile_publisher import (
    NetworkProfilePublisher,
    NetworkProfileWriteError,
)
from worker.domain.profiling.network_profile import NetworkProfile


class RedisNetworkProfilePublisher(NetworkProfilePublisher):
    def __init__(
        self,
        worker_id: WorkerId,
        redis: redis_asyncio.Redis,
        network_profile_key_factory_cls: type[NetworkProfileKeyFactory],
    ) -> None:
        self._worker_id = worker_id
        self._redis = redis
        self._network_profile_key_factory_cls = network_profile_key_factory_cls

    @override
    async def publish_network_profile(self, network_profile: NetworkProfile) -> None:
        try:
            redis_key = await self._network_profile_key_factory_cls.build_network_profile_key_per_worker(
                self._worker_id
            )
            dto = NetworkProfileMapper.network_profile_to_dto(
                self._worker_id,
                network_profile,
            )
            redis_value = dto.model_dump_json()

            success = await self._redis.set(redis_key, redis_value)
            if not success:
                raise NetworkProfileWriteError("Failed to publish network profile")
        except (ValueError, RedisError) as e:
            raise NetworkProfileWriteError("Unable to publish network profile") from e
