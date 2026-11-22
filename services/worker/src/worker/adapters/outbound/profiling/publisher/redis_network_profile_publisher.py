from typing import override

from redis import RedisError
from redis import asyncio as redis_asyncio

from redis_access.address_factory.worker_profiles.network_profile_address_factory import (
    NetworkProfileAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
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
        network_profile_address_factory_cls: type[
            NetworkProfileAddressFactory[HashMapRedisAddress]
        ],
    ) -> None:
        self._worker_id = worker_id
        self._redis = redis
        self._network_profile_address_factory_cls = network_profile_address_factory_cls

    @override
    async def publish_network_profile(self, network_profile: NetworkProfile) -> None:
        try:
            addr = self._network_profile_address_factory_cls.build_worker_network_profile_address(
                self._worker_id
            )
            dto = NetworkProfileMapper.network_profile_to_dto(
                self._worker_id,
                network_profile,
            )
            redis_value = dto.model_dump_json()

            await self._redis.hset(addr.hash_map_key, addr.field_key, redis_value)

        except (ValueError, RedisError) as e:
            raise NetworkProfileWriteError("Unable to publish network profile") from e
