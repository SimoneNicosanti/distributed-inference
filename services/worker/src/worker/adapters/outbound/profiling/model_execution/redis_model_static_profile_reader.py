from typing import override

from redis import asyncio as redis_asyncio

from integration.redis.profiled_model_variant.keys_factory import (
    build_profiled_model_variant_dict_name,
    build_profiled_model_variant_ids_key,
    build_profiled_model_variant_key,
)
from integration.redis.profiled_model_variant.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
    ShapePointDto,
)
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.application.ports.outbound.profiling.model_execution.model_static_profile_reader import (
    ModelStaticProfileReader,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class RedisModelStaticProfileReader(ModelStaticProfileReader):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_all_model_static_profile_ids(self) -> list[ModelVariantId]:
        ids_key = build_profiled_model_variant_ids_key()

        return [
            ModelVariantId.model_validate_json(value)
            async for value in self._redis.sscan_iter(ids_key)
        ]

    @override
    async def get_model_static_profile(
        self,
        model_variant_id: ModelVariantId,
    ) -> ModelStaticProfile:
        dict_name = await build_profiled_model_variant_dict_name(model_variant_id)
        key = await build_profiled_model_variant_key(model_variant_id)

        value = await self._redis.hget(dict_name, key)
        if value is None:
            raise ValueError(f"Model static profile {model_variant_id} not found")

        dto = ProfiledModelVariantDto.model_validate_json(value)
        return self._build_model_static_profile_from_dto(dto)

    @staticmethod
    def _build_model_static_profile_from_dto(
        dto: ProfiledModelVariantDto,
    ) -> ModelStaticProfile:

        model_variant_id = dto.model_variant_id
        artifact_ref = dto.artifact_ref

        shape_point_dtos: list[ShapePointDto] = dto.profile.shape_points

        shape_points = []
        for shape_point_dto in shape_point_dtos:
            input_shape_points = {
                input_shape_point.name: input_shape_point.shape
                for input_shape_point in shape_point_dto.input_shape_points
            }

            shape_points.append(ShapePoint(input_shape_points=input_shape_points))

        contractions: list[tuple[LayerKey, ...]] = dto.profile.optimization_contractions

        return ModelStaticProfile(
            model_variant_id=model_variant_id,
            artifact_ref=artifact_ref,
            shape_points=shape_points,
            contractions=contractions,
        )
