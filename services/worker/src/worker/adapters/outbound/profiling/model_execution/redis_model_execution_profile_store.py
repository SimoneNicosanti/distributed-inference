from typing import override

from redis import asyncio as redis_asyncio

from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from integration.redis.worker_profiles.execution_profile_keys import (
    ExecutionProfileKeyFactory,
)
from integration.shared.id_json import (
    model_variant_id_from_json,
    model_variant_id_to_json,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.adapters.outbound.profiling.model_execution.mappers.model_execution_profile_mapper import (
    ModelExecutionProfileMapper,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfilePublisher,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_reader import (
    ModelExecutionProfileReader,
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
        execution_profile_key_factory_cls: type[ExecutionProfileKeyFactory],
    ) -> None:
        self._redis = redis
        self._execution_profile_key_factory_cls = execution_profile_key_factory_cls

    @override
    async def puplish_model_execution_profile(
        self, worker_id: WorkerId, model_execution_profile: ModelExecutionProfile
    ) -> None:
        worker_key = await self._execution_profile_key_factory_cls.build_execution_profile_key_per_worker(
            worker_id
        )
        worker_ids_key = await self._execution_profile_key_factory_cls.build_execution_profile_ids_key_per_worker(
            worker_id
        )
        profile_id = await self._execution_profile_key_factory_cls.build_model_execution_profile_key(
            model_execution_profile.model_version_id
        )
        profile_json = ModelExecutionProfileMapper.model_execution_profile_to_dto(
            worker_id,
            model_execution_profile,
        ).model_dump_json()
        model_variant_id_json = model_variant_id_to_json(
            model_execution_profile.model_version_id
        )

        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(worker_key, profile_id, profile_json)
            pipeline.sadd(worker_ids_key, model_variant_id_json)
            await pipeline.execute()

    @override
    async def get_model_execution_profile_by_worker_id(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> ModelExecutionProfile:
        worker_key = await self._execution_profile_key_factory_cls.build_execution_profile_key_per_worker(
            worker_id
        )
        profile_id = await self._execution_profile_key_factory_cls.build_model_execution_profile_key(
            model_version_id
        )

        value = await self._redis.hget(worker_key, profile_id)

        if value is None:
            raise ValueError("Model execution profile not found")

        dto = ModelExecutionProfileDto.model_validate_json(value)
        ModelExecutionProfileMapper.validate_dto_identity(
            dto, worker_id, model_version_id
        )
        return ModelExecutionProfileMapper.model_execution_profile_from_dto(dto)

    @override
    async def get_all_profiled_model_ids_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelVariantId]:
        return [
            model_variant_id_from_json(value)
            async for value in self._redis.sscan_iter(
                await self._execution_profile_key_factory_cls.build_execution_profile_ids_key_per_worker(
                    worker_id
                )
            )
        ]

    @override
    async def get_all_profiles_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelExecutionProfile]:
        worker_key = await self._execution_profile_key_factory_cls.build_execution_profile_key_per_worker(
            worker_id
        )

        values = await self._redis.hvals(worker_key)

        profiles: list[ModelExecutionProfile] = []
        for value in values:
            dto = ModelExecutionProfileDto.model_validate_json(value)
            ModelExecutionProfileMapper.validate_dto_identity(dto, worker_id)
            profiles.append(
                ModelExecutionProfileMapper.model_execution_profile_from_dto(dto)
            )
        return profiles

    @override
    async def check_model_execution_profile_exists(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> bool:
        worker_key = await self._execution_profile_key_factory_cls.build_execution_profile_key_per_worker(
            worker_id
        )
        profile_id = await self._execution_profile_key_factory_cls.build_model_execution_profile_key(
            model_version_id
        )

        return await self._redis.hexists(worker_key, profile_id)
