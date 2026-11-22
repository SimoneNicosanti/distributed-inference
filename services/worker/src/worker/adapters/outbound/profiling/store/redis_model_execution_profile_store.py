from typing import override

from redis import RedisError
from redis import asyncio as redis_asyncio

from integration.shared.id_json import model_variant_id_from_json
from redis_access.address_factory.worker_profiles.execution_profile_address_factory import (
    ExecutionProfileAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.redis_reader.abc.hash_fields_redis_reader import HashFieldsRedisReader
from redis_access.redis_reader.abc.redis_existence_checker import RedisExistenceChecker
from redis_access.redis_reader.abc.single_value_redis_reader import (
    RedisReadError,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.store.mappers.model_execution_profile_mapper import (
    ModelExecutionProfileMapper,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfilePublisher,
    ModelExecutionProfileWriteError,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_reader import (
    ModelExecutionProfileReader,
    ModelExecutionProfileReadError,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)


class RedisModelExecutionProfileStore(
    ModelExecutionProfilePublisher, ModelExecutionProfileReader
):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        execution_profile_address_factory_cls: type[
            ExecutionProfileAddressFactory[HashMapRedisAddress]
        ],
        fields_reader: HashFieldsRedisReader,
        existence_checker: RedisExistenceChecker[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        self._execution_profile_address_factory_cls = (
            execution_profile_address_factory_cls
        )
        self._fields_reader = fields_reader
        self._existence_checker = existence_checker

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

    @override
    async def get_all_profiled_model_ids_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelVariantId]:
        try:
            addr = await self._execution_profile_address_factory_cls.build_worker_execution_profiles_address(
                worker_id
            )
            fields = await self._fields_reader.read_fields(self._redis, addr)
            return sorted(
                (model_variant_id_from_json(field) for field in fields),
                key=lambda variant: (variant.model_id.model_name, variant.variant_tag),
            )
        except (ValueError, RedisReadError) as e:
            raise ModelExecutionProfileReadError(
                f"Unable to read model execution profile IDs: worker={worker_id}"
            ) from e

    @override
    async def check_model_execution_profile_exists(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> bool:
        try:
            addr = await self._execution_profile_address_factory_cls.build_worker_model_execution_profile_address(
                worker_id, model_version_id
            )
            return await self._existence_checker.exists(self._redis, addr)
        except (ValueError, RedisReadError) as e:
            raise ModelExecutionProfileReadError(
                f"Unable to check execution profile existence: worker={worker_id}, model={model_version_id}"
            ) from e
