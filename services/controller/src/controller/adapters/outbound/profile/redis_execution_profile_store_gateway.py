from collections.abc import Iterable
from typing import override

from redis import asyncio as redis_asyncio

from controller.adapters.outbound.profile.mappers.model_variant_execution_profile_mapper import (
    ModelVariantExecutionProfileMapper,
)
from controller.application.ports.outbound.profile.execution_profile_store_gateway import (
    ExecutionProfileStoreGateway,
)
from controller.domain.profile.model_variant_execution_profile import (
    ModelVariantExecutionProfile,
    WorkerExecutionProfiles,
)
from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from integration.redis.worker_profiles.execution_profile_keys import (
    ExecutionProfileKeyFactory,
)
from integration.shared.id_json import model_variant_id_to_json
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class RedisExecutionProfileStoreGateway(ExecutionProfileStoreGateway):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        execution_profile_key_factory_cls: type[ExecutionProfileKeyFactory],
    ) -> None:
        self._redis = redis
        self._execution_profile_key_factory_cls = execution_profile_key_factory_cls

    @override
    async def get_execution_profiles_for_workers_and_variants(
        self,
        worker_ids: Iterable[WorkerId],
        model_variant_ids: Iterable[ModelVariantId],
    ) -> list[WorkerExecutionProfiles]:
        ordered_worker_ids = sorted(
            set(worker_ids),
            key=lambda worker_id: str(worker_id.id),
        )
        ordered_model_variant_ids = sorted(
            set(model_variant_ids),
            key=model_variant_id_to_json,
        )
        requested_model_variant_ids = frozenset(ordered_model_variant_ids)

        if not ordered_worker_ids:
            return []
        if not ordered_model_variant_ids:
            return [
                WorkerExecutionProfiles(worker_id=worker_id, execution_profiles={})
                for worker_id in ordered_worker_ids
            ]

        async with self._redis.pipeline(transaction=False) as pipeline:
            for worker_id in ordered_worker_ids:
                pipeline.hvals(
                    await self._execution_profile_key_factory_cls.build_execution_profile_key_per_worker(
                        worker_id
                    )
                )
            stored_profiles_by_worker = await pipeline.execute()

        return [
            self._build_worker_profiles(
                worker_id,
                stored_profiles,
                ordered_model_variant_ids,
                requested_model_variant_ids,
            )
            for worker_id, stored_profiles in zip(
                ordered_worker_ids,
                stored_profiles_by_worker,
                strict=True,
            )
        ]

    @classmethod
    def _build_worker_profiles(
        cls,
        worker_id: WorkerId,
        stored_profiles: Iterable[str | bytes | bytearray],
        ordered_model_variant_ids: list[ModelVariantId],
        requested_model_variant_ids: frozenset[ModelVariantId],
    ) -> WorkerExecutionProfiles:
        profiles_by_variant: dict[
            ModelVariantId,
            ModelVariantExecutionProfile,
        ] = {}

        for value in stored_profiles:
            dto = ModelExecutionProfileDto.model_validate_json(value)
            if dto.worker_id != worker_id:
                raise ValueError(
                    "Stored execution profile worker does not match the requested worker"
                )
            if dto.model_variant_id not in requested_model_variant_ids:
                continue
            if dto.model_variant_id in profiles_by_variant:
                raise ValueError(
                    "Duplicate execution profiles for a worker and model variant"
                )

            profiles_by_variant[dto.model_variant_id] = (
                ModelVariantExecutionProfileMapper.model_variant_execution_profile_from_dto(
                    dto
                )
            )

        return WorkerExecutionProfiles(
            worker_id=worker_id,
            execution_profiles={
                model_variant_id: profiles_by_variant[model_variant_id]
                for model_variant_id in ordered_model_variant_ids
                if model_variant_id in profiles_by_variant
            },
        )
