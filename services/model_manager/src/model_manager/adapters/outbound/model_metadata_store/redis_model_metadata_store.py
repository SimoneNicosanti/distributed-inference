from typing import override

from redis import asyncio as redis_asyncio

from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
)
from integration.redis.model.keys_factory import ModelKeyFactory
from integration.redis.model_partition.keys_factory import (
    ModelPartitionKeyFactory,
)
from integration.redis.profiled_model_variant.keys_factory import (
    ProfiledModelVariantKeyFactory,
)
from integration.shared.id_json import model_variant_id_to_json
from model_manager.adapters.outbound.model_metadata_store.mappers.model_mapper import (
    ModelMapper,
)
from model_manager.adapters.outbound.model_metadata_store.mappers.model_partition_mapper import (
    ModelPartitionMapper,
)
from model_manager.adapters.outbound.model_metadata_store.mappers.profiled_model_variant_mapper import (
    ProfiledModelVariantMapper,
)
from model_manager.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from model_manager.domain.model import Model
from model_manager.domain.model_partition import ModelPartition
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class RedisModelMetadataStore(ModelMetadataStore):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        model_key_factory_cls: type[ModelKeyFactory],
        profiled_model_variant_key_factory_cls: type[ProfiledModelVariantKeyFactory],
        model_partition_key_factory_cls: type[ModelPartitionKeyFactory],
    ) -> None:
        self._redis = redis
        self._model_key_factory_cls = model_key_factory_cls
        self._profiled_model_variant_key_factory_cls = (
            profiled_model_variant_key_factory_cls
        )
        self._model_partition_key_factory_cls = model_partition_key_factory_cls

    @override
    async def add_model(self, model: Model) -> None:
        model_key = await self._model_key_factory_cls.build_model_key(model.model_id)

        dto = ModelMapper.to_model_dto(model)
        await self._redis.set(model_key, dto.model_dump_json())

    @override
    async def get_model(self, model_id: ModelId) -> Model:
        model_key = await self._model_key_factory_cls.build_model_key(model_id)

        value = await self._redis.get(model_key)
        if value is None:
            raise ValueError(f"Model {model_id} not found")

        dto = ModelDto.model_validate_json(value)
        return ModelMapper.from_model_dto(dto)

    @override
    async def check_model_existence(self, model_id: ModelId) -> bool:
        model_key = await self._model_key_factory_cls.build_model_key(model_id)

        num_values = await self._redis.exists(model_key)
        return num_values > 0

    @override
    async def add_profiled_model_variant(
        self,
        profiled_model_variant: ProfiledModelVariant,
    ) -> None:
        set_name = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_dict_name(
            profiled_model_variant.model_variant_id
        )
        profiled_model_variant_key = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_key(
            profiled_model_variant.model_variant_id
        )
        dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(
            profiled_model_variant
        )
        dto_json = dto.model_dump_json()
        model_variant_id_json = model_variant_id_to_json(
            profiled_model_variant.model_variant_id
        )

        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(
                set_name,
                profiled_model_variant_key,
                dto_json,
            )
            pipeline.sadd(
                self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_ids_key(),
                model_variant_id_json,
            )
            await pipeline.execute()

    @override
    async def get_profiled_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> ProfiledModelVariant:
        set_name = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_dict_name(
            model_variant_id
        )
        profiled_model_variant_key = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_key(
            model_variant_id
        )

        value = await self._redis.hget(set_name, profiled_model_variant_key)

        if value is None:
            raise ValueError(f"Profiled model variant {model_variant_id} not found")

        dto = ProfiledModelVariantDto.model_validate_json(value)
        return ProfiledModelVariantMapper.from_profiled_model_variant_dto(dto)

    @override
    async def check_profiled_model_variant_existence(
        self,
        model_variant_id: ModelVariantId,
    ) -> bool:
        set_name = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_dict_name(
            model_variant_id
        )
        profiled_model_variant_key = await self._profiled_model_variant_key_factory_cls.build_profiled_model_variant_key(
            model_variant_id
        )

        exists = await self._redis.hexists(set_name, profiled_model_variant_key)
        return exists

    @override
    async def add_model_partition(
        self,
        model_partition: ModelPartition,
    ) -> None:
        set_name = (
            await self._model_partition_key_factory_cls.build_model_partition_dict_name(
                model_partition.model_partition_id
            )
        )
        partition_key = (
            await self._model_partition_key_factory_cls.build_model_partition_key(
                model_partition.model_partition_id
            )
        )

        dto = ModelPartitionMapper.to_model_partition_dto(model_partition)
        dto_json = dto.model_dump_json()

        await self._redis.hset(
            set_name,
            partition_key,
            dto_json,
        )
