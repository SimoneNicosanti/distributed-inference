from typing import override

from redis import RedisError
from redis import asyncio as redis_asyncio

from redis_access.address_factory.worker_profiles.execution_profile_address_factory import (
    ExecutionProfileAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.publisher.mappers.model_execution_profile_mapper import (
    ModelExecutionProfileMapper,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfilePublisher,
    ModelExecutionProfileWriteError,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)


class RedisModelExecutionProfilePublisher(ModelExecutionProfilePublisher):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        execution_profile_address_factory_cls: type[
            ExecutionProfileAddressFactory[HashMapRedisAddress]
        ],
    ) -> None:
        self._redis = redis
        self._execution_profile_address_factory_cls = (
            execution_profile_address_factory_cls
        )

    @override
    async def puplish_model_execution_profile(
        self, worker_id: WorkerId, model_execution_profile: ModelExecutionProfile
    ) -> None:
        try:
            variant_id = model_execution_profile.model_version_id
            addr = await self._execution_profile_address_factory_cls.build_worker_model_execution_profile_address(
                worker_id, variant_id
            )
            profile_json = ModelExecutionProfileMapper.model_execution_profile_to_dto(
                worker_id, model_execution_profile
            ).model_dump_json()

            await self._redis.hset(addr.hash_map_key, addr.field_key, profile_json)
        except (ValueError, RedisError) as e:
            raise ModelExecutionProfileWriteError(
                f"Unable to publish model execution profile: worker={worker_id}, model={model_execution_profile.model_version_id}"
            ) from e
