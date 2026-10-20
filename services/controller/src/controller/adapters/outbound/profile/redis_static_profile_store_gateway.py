import asyncio
from typing import override

from pydantic import BaseModel, ConfigDict
from redis import asyncio as redis_asyncio

from controller.application.ports.outbound.profile.static_profile_store_gateway import (
    StaticProfileStoreGateway,
)
from controller.domain.profile.model_variant_static_profile import (
    LayerProperty,
    ModelVariantInfo,
    ModelVariantStaticProfile,
    ShapeProfile,
    TensorProperty,
)
from controller.domain.profile.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from integration.redis.model.keys_factory import build_model_key
from integration.redis.profiled_model_variant.keys_factory import (
    build_profiled_model_variant_dict_name,
    build_profiled_model_variant_ids_key,
    build_profiled_model_variant_key,
)
from integration.redis.profiled_model_variant.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
)
from shared.model.model import ModelId, ModelTask, ModelVisibility
from shared.model.model_variant import ModelVariantId


class _StoredModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_task: ModelTask


class _StoredModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: ModelId
    visibility: ModelVisibility
    model_info: _StoredModelInfo


class RedisStaticProfileStoreGateway(StaticProfileStoreGateway):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_all_model_variant_static_profiles(
        self,
    ) -> list[ModelVariantStaticProfile]:
        ids_key = build_profiled_model_variant_ids_key()
        model_variant_ids = [
            ModelVariantId.model_validate_json(value)
            async for value in self._redis.sscan_iter(ids_key)
        ]
        model_variant_ids.sort(
            key=lambda model_variant_id: model_variant_id.model_dump_json()
        )
        if not model_variant_ids:
            return []

        profiled_model_variants = await asyncio.gather(
            *(
                self._get_profiled_model_variant(model_variant_id)
                for model_variant_id in model_variant_ids
            )
        )

        model_ids = sorted(
            {dto.model_variant_id.model_id for dto in profiled_model_variants},
            key=lambda model_id: model_id.model_dump_json(),
        )
        models = await asyncio.gather(
            *(self._get_model(model_id) for model_id in model_ids)
        )
        models_by_id = {model.model_id: model for model in models}

        return [
            self._to_static_profile(dto, models_by_id[dto.model_variant_id.model_id])
            for dto in profiled_model_variants
        ]

    async def _get_profiled_model_variant(
        self,
        model_variant_id: ModelVariantId,
    ) -> ProfiledModelVariantDto:
        dict_name, key = await asyncio.gather(
            build_profiled_model_variant_dict_name(model_variant_id),
            build_profiled_model_variant_key(model_variant_id),
        )
        value = await self._redis.hget(dict_name, key)
        if value is None:
            raise ValueError(f"Profiled model variant {model_variant_id} not found")

        dto = ProfiledModelVariantDto.model_validate_json(value)
        if dto.model_variant_id != model_variant_id:
            raise ValueError(
                "Stored profiled model variant ID does not match its Redis index"
            )
        return dto

    async def _get_model(self, model_id: ModelId) -> _StoredModel:
        key = await build_model_key(model_id)
        value = await self._redis.get(key)
        if value is None:
            raise ValueError(f"Model {model_id} not found")

        model = _StoredModel.model_validate_json(value)
        if model.model_id != model_id:
            raise ValueError("Stored model ID does not match its Redis key")
        return model

    @staticmethod
    def _to_static_profile(
        dto: ProfiledModelVariantDto,
        model: _StoredModel,
    ) -> ModelVariantStaticProfile:
        profile = dto.profile

        return ModelVariantStaticProfile(
            model_variant_id=dto.model_variant_id,
            info=ModelVariantInfo(
                visibility=model.visibility,
                task=model.model_info.model_task,
                accuracies=tuple(dto.info.accuracies),
            ),
            topology=ModelVariantTopology(
                layers={
                    layer_key: LayerDescription.model_validate(description.model_dump())
                    for layer_key, description in profile.topology.layers.items()
                },
                edges=(
                    EdgeDescription.model_validate(description.model_dump())
                    for description in profile.topology.edges
                ),
            ),
            invariant_layer_properties={
                layer_key: LayerProperty.model_validate(property_.model_dump())
                for layer_key, property_ in profile.invariant_layer_properties.items()
            },
            invariant_tensor_properties={
                tensor_key: TensorProperty.model_validate(property_.model_dump())
                for tensor_key, property_ in (
                    profile.invariant_tensor_properties.items()
                )
            },
            shape_profiles=tuple(
                ShapeProfile.model_validate(shape_profile.model_dump())
                for shape_profile in profile.shape_profiles
            ),
            optimization_contractions=tuple(profile.optimization_contractions),
        )
