from typing import override

from redis import asyncio as redis_asyncio

from integration.dto.model.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
)
from integration.redis.profiled_model_variant.keys_factory import (
    ProfiledModelVariantKeyFactory,
)
from integration.shared.id_json import model_variant_id_from_json
from shared.model.model_variant import ModelVariantId
from worker.adapters.outbound.profiling.model_execution.mappers.model_static_profile_mapper import (
    ModelStaticProfileMapper,
)
from worker.application.ports.outbound.profiling.model_execution.model_static_profile_reader import (
    ModelStaticProfileReader,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)


class RedisModelStaticProfileReader(ModelStaticProfileReader):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        profiled_model_variant_key_factory_cls: type[ProfiledModelVariantKeyFactory],
    ) -> None:
        self._redis = redis
        self._profiled_model_variant_key_factory_cls = (
            profiled_model_variant_key_factory_cls
        )

    @override
    async def get_all_model_static_profile_ids(self) -> list[ModelVariantId]:
        ids_key = self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_ids_key()

        return [
            model_variant_id_from_json(value)
            async for value in self._redis.sscan_iter(ids_key)
        ]

    @override
    async def get_model_static_profile(
        self,
        model_variant_id: ModelVariantId,
    ) -> ModelStaticProfile:
        dict_name = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_dict_name(
            model_variant_id
        )
        key = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_key(
            model_variant_id
        )

        value = await self._redis.hget(dict_name, key)
        if value is None:
            raise ValueError(f"Model static profile {model_variant_id} not found")

        dto = ProfiledModelVariantDto.model_validate_json(value)
        return ModelStaticProfileMapper.model_static_profile_from_dto(dto)
