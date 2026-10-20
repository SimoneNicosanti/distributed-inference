from collections.abc import Iterable
from typing import override

from redis import asyncio as redis_asyncio

from controller.application.ports.outbound.profile.execution_profile_store_gateway import (
    ExecutionProfileStoreGateway,
)
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
    ModelVariantExecutionProfile,
    ShapeExecutionProfile,
    WorkerExecutionProfiles,
)
from controller.domain.profile.model_variant_static_profile import (
    InputShapePoint,
    ShapePoint,
)
from integration.worker.profiling.redis.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from integration.worker.profiling.redis.execution_profile_keys import (
    build_execution_profile_key_per_worker,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class RedisExecutionProfileStoreGateway(ExecutionProfileStoreGateway):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_execution_profiles_for_workers_and_variants(
        self,
        worker_ids: Iterable[WorkerId],
        model_variant_ids: Iterable[ModelVariantId],
    ) -> list[WorkerExecutionProfiles]:
        ordered_worker_ids = sorted(
            set(worker_ids),
            key=lambda worker_id: str(worker_id.service_id),
        )
        ordered_model_variant_ids = sorted(
            set(model_variant_ids),
            key=lambda model_variant_id: model_variant_id.model_dump_json(),
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
                pipeline.hvals(build_execution_profile_key_per_worker(worker_id))
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

            profiles_by_variant[dto.model_variant_id] = cls._to_domain(dto)

        return WorkerExecutionProfiles(
            worker_id=worker_id,
            execution_profiles={
                model_variant_id: profiles_by_variant[model_variant_id]
                for model_variant_id in ordered_model_variant_ids
                if model_variant_id in profiles_by_variant
            },
        )

    @staticmethod
    def _to_domain(
        dto: ModelExecutionProfileDto,
    ) -> ModelVariantExecutionProfile:
        shape_execution_profiles: list[ShapeExecutionProfile] = []

        for shape_profile in dto.shape_execution_profiles:
            layer_profiles: dict[str, LayerExecutionProfile] = {}
            for layer_key, layer_profile in shape_profile.layer_profiles.items():
                if layer_profile.layer_key != layer_key:
                    raise ValueError(
                        "Stored execution profile layer key does not match its index"
                    )
                layer_profiles[layer_key] = LayerExecutionProfile(
                    execution_time_s=(
                        layer_profile.cpu_execution_profile.execution_time
                    )
                )

            shape_execution_profiles.append(
                ShapeExecutionProfile(
                    shape_point=ShapePoint(
                        input_shape_points=tuple(
                            InputShapePoint(name=name, shape=shape)
                            for name, shape in sorted(
                                shape_profile.shape_point.input_shape_points.items()
                            )
                        )
                    ),
                    layer_profiles=layer_profiles,
                )
            )

        return ModelVariantExecutionProfile(
            shape_execution_profiles=tuple(shape_execution_profiles)
        )
