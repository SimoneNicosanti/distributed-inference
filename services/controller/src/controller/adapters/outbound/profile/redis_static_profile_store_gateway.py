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
from integration.shared.id_json import (
    model_id_to_json,
    model_variant_id_from_json,
    model_variant_id_to_json,
)
from redis_access.address_factory.model.address_factory import ModelAddressFactory
from redis_access.address_factory.profiled_model_variant.address_factory import (
    ProfiledModelVariantAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
from redis_access.redis_reader.abc.multi_value_redis_reader import MultiValueRedisReader
from redis_access.redis_reader.abc.single_value_redis_reader import (
    RedisValueNotFoundError,
    SingleValueRedisReader,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class RedisStaticProfileStoreGateway(StaticProfileStoreGateway):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        model_address_factory_cls: type[ModelAddressFactory[KeyRedisAddress]],
        profiled_model_variant_address_factory_cls: type[
            ProfiledModelVariantAddressFactory[HashMapRedisAddress]
        ],
        model_reader: SingleValueRedisReader[KeyRedisAddress],
        variant_reader: SingleValueRedisReader[HashMapRedisAddress],
        ids_reader: MultiValueRedisReader[KeyRedisAddress],
    ) -> None:
        self._redis = redis
        self._model_reader = model_reader
        self._variant_reader = variant_reader
        self._ids_reader = ids_reader
        self._model_address_factory_class = model_address_factory_cls
        self._profiled_model_variant_address_factory_cls = (
            profiled_model_variant_address_factory_cls
        )

    @override
    async def get_all_model_variant_static_profiles(
        self,
    ) -> list[ModelVariantStaticProfile]:
        ids_address = self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_ids_address()
        model_variant_ids = [
            model_variant_id_from_json(value)
            for value in await self._ids_reader.read_all(self._redis, ids_address)
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
        addr = await self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_address(
            model_variant_id
        )
        try:
            value = await self._variant_reader.read_single(self._redis, addr)
        except RedisValueNotFoundError as e:
            raise ValueError(
                f"Profiled model variant {model_variant_id} not found"
            ) from e

        dto = ProfiledModelVariantDto.model_validate_json(value)
        if dto.model_variant_id != model_variant_id:
            raise ValueError(
                "Stored profiled model variant ID does not match its Redis index"
            )
        return dto

    async def _get_model(self, model_id: ModelId) -> ModelDto:
        key = await self._model_address_factory_class.build_model_address(model_id)
        try:
            value = await self._model_reader.read_single(self._redis, key)
        except RedisValueNotFoundError as e:
            raise ValueError(f"Model {model_id} not found") from e

        model = ModelDto.model_validate_json(value)
        if model.model_id != model_id:
            raise ValueError("Stored model ID does not match its Redis key")
        return model
