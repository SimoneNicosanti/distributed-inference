from typing import override

from redis import asyncio as redis_asyncio

from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
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
from redis_access.address_factory.model.address_factory import ModelAddressFactory
from redis_access.address_factory.model_partition.address_factory import (
    ModelPartitionAddressFactory,
)
from redis_access.address_factory.profiled_model_variant.address_factory import (
    ProfiledModelVariantAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
from redis_access.redis_reader.abc.redis_reader import (
    RedisReader,
    RedisValueNotFoundError,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class RedisModelMetadataStore(ModelMetadataStore):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        model_address_factory_cls: type[ModelAddressFactory[KeyRedisAddress]],
        profiled_model_variant_address_factory_cls: type[
            ProfiledModelVariantAddressFactory[HashMapRedisAddress]
        ],
        model_partition_address_factory_cls: type[
            ModelPartitionAddressFactory[HashMapRedisAddress]
        ],
        model_reader: RedisReader[KeyRedisAddress],
        variant_reader: RedisReader[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        self._model_address_factory_cls = model_address_factory_cls
        self._profiled_model_variant_address_factory_cls = (
            profiled_model_variant_address_factory_cls
        )
        self._model_partition_address_factory_cls = model_partition_address_factory_cls
        self._model_reader = model_reader
        self._variant_reader = variant_reader

    @override
    async def add_model(self, model: Model) -> None:
        model_key = await self._model_address_factory_cls.build_model_address(
            model.model_id
        )

        dto = ModelMapper.to_model_dto(model)
        await self._redis.set(model_key.key, dto.model_dump_json())

    @override
    async def get_model(self, model_id: ModelId) -> Model:
        model_key = await self._model_address_factory_cls.build_model_address(model_id)

        try:
            value = await self._model_reader.read(self._redis, model_key)
        except RedisValueNotFoundError as e:
            raise ValueError(f"Model {model_id} not found") from e

        dto = ModelDto.model_validate_json(value)
        if dto.model_id != model_id:
            raise ValueError("Stored model ID mismatch")
        return ModelMapper.from_model_dto(dto)

    @override
    async def check_model_existence(self, model_id: ModelId) -> bool:
        model_key = await self._model_address_factory_cls.build_model_address(model_id)

        num_values = await self._redis.exists(model_key.key)
        return num_values > 0

    @override
    async def add_profiled_model_variant(
        self,
        profiled_model_variant: ProfiledModelVariant,
    ) -> None:
        addr = await self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_address(
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
                addr.hash_map_key,
                addr.field_key,
                dto_json,
            )
            pipeline.sadd(
                self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_ids_address().key,
                model_variant_id_json,
            )
            await pipeline.execute()

    @override
    async def get_profiled_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> ProfiledModelVariant:
        addr = await self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_address(
            model_variant_id
        )
        try:
            value = await self._variant_reader.read(self._redis, addr)
        except RedisValueNotFoundError as e:
            raise ValueError(
                f"Profiled model variant {model_variant_id} not found"
            ) from e

        dto = ProfiledModelVariantDto.model_validate_json(value)
        if dto.model_variant_id != model_variant_id:
            raise ValueError("Stored model variant ID mismatch")
        return ProfiledModelVariantMapper.from_profiled_model_variant_dto(dto)

    @override
    async def check_profiled_model_variant_existence(
        self,
        model_variant_id: ModelVariantId,
    ) -> bool:
        addr = await self._profiled_model_variant_address_factory_cls.build_profiled_model_variant_address(
            model_variant_id
        )
        exists = await self._redis.hexists(addr.hash_map_key, addr.field_key)
        return exists

    @override
    async def add_model_partition(
        self,
        model_partition: ModelPartition,
    ) -> None:
        addr = await self._model_partition_address_factory_cls.build_model_partition_address(
            model_partition.model_partition_id
        )

        dto = ModelPartitionMapper.to_model_partition_dto(model_partition)
        dto_json = dto.model_dump_json()

        await self._redis.hset(
            addr.hash_map_key,
            addr.field_key,
            dto_json,
        )
