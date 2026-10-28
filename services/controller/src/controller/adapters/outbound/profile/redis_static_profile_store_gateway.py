import asyncio
from typing import override

from redis import asyncio as redis_asyncio

from controller.adapters.outbound.profile.mappers.model_variant_static_profile_mapper import (
    ModelVariantStaticProfileMapper,
)
from controller.application.ports.outbound.profile.static_profile_store_gateway import (
    StaticProfileStoreGateway,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
)
from integration.redis.model.keys_factory import ModelKeyFactory
from integration.redis.profiled_model_variant.keys_factory import (
    ProfiledModelVariantKeyFactory,
)
from integration.shared.id_json import (
    model_id_to_json,
    model_variant_id_from_json,
    model_variant_id_to_json,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class RedisStaticProfileStoreGateway(StaticProfileStoreGateway):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        model_key_factory_cls: type[ModelKeyFactory],
        profiled_model_variant_key_factory_cls: type[ProfiledModelVariantKeyFactory],
    ) -> None:
        self._redis = redis
        self._model_key_factory_class = model_key_factory_cls
        self._profiled_model_variant_key_factory_cls = (
            profiled_model_variant_key_factory_cls
        )

    @override
    async def get_all_model_variant_static_profiles(
        self,
    ) -> list[ModelVariantStaticProfile]:
        ids_key = self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_ids_key()
        model_variant_ids = [
            model_variant_id_from_json(value)
            async for value in self._redis.sscan_iter(ids_key)
        ]
        model_variant_ids.sort(key=model_variant_id_to_json)
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
            key=model_id_to_json,
        )
        models = await asyncio.gather(
            *(self._get_model(model_id) for model_id in model_ids)
        )
        models_by_id = {model.model_id: model for model in models}

        return [
            ModelVariantStaticProfileMapper.model_variant_static_profile_from_dto(
                dto, models_by_id[dto.model_variant_id.model_id]
            )
            for dto in profiled_model_variants
        ]

    async def _get_profiled_model_variant(
        self,
        model_variant_id: ModelVariantId,
    ) -> ProfiledModelVariantDto:
        dict_name, key = await asyncio.gather(
            self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_dict_name(
                model_variant_id
            ),
            self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_key(
                model_variant_id
            ),
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

    async def _get_model(self, model_id: ModelId) -> ModelDto:
        key = await self._model_key_factory_class.build_model_key(model_id)
        value = await self._redis.get(key)
        if value is None:
            raise ValueError(f"Model {model_id} not found")

        model = ModelDto.model_validate_json(value)
        if model.model_id != model_id:
            raise ValueError("Stored model ID does not match its Redis key")
        return model
