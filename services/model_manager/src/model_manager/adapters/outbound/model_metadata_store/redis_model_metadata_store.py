from typing import override

from redis import asyncio as redis_asyncio

from integration.redis.model.keys_factory import build_model_key
from integration.redis.model_partition.keys_factory import (
    build_model_partition_dict_name,
    build_model_partition_key,
)
from integration.redis.profiled_model_variant.keys_factory import (
    build_profiled_model_variant_dict_name,
    build_profiled_model_variant_key,
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
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def add_model(self, model: Model) -> None:
        model_key = await build_model_key(model.model_id)

        await self._redis.set(model_key, model.model_dump_json())

    @override
    async def get_model(self, model_id: ModelId) -> Model:
        model_key = await build_model_key(model_id)

        value = await self._redis.get(model_key)
        if value is None:
            raise ValueError(f"Model {model_id} not found")

        return Model.model_validate_json(value)

    @override
    async def check_model_existence(self, model_id: ModelId) -> bool:
        model_key = await build_model_key(model_id)

        num_values = await self._redis.exists(model_key)
        return num_values > 0

    @override
    async def add_profiled_model_variant(
        self,
        profiled_model_variant: ProfiledModelVariant,
    ) -> None:
        set_name = await build_profiled_model_variant_dict_name(
            profiled_model_variant.model_variant_id
        )
        profiled_model_variant_key = await build_profiled_model_variant_key(
            profiled_model_variant.model_variant_id
        )

        await self._redis.hset(
            set_name,
            profiled_model_variant_key,
            profiled_model_variant.model_dump_json(),
        )
        pass

    @override
    async def get_profiled_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> ProfiledModelVariant:
        set_name = await build_profiled_model_variant_dict_name(model_variant_id)
        profiled_model_variant_key = await build_profiled_model_variant_key(
            model_variant_id
        )

        value = await self._redis.hget(set_name, profiled_model_variant_key)

        if value is None:
            raise ValueError(f"Profiled model variant {model_variant_id} not found")

        return ProfiledModelVariant.model_validate_json(value)

    @override
    async def check_profiled_model_variant_existence(
        self,
        model_variant_id: ModelVariantId,
    ) -> bool:
        set_name = await build_profiled_model_variant_dict_name(model_variant_id)
        profiled_model_variant_key = await build_profiled_model_variant_key(
            model_variant_id
        )

        exists = await self._redis.hexists(set_name, profiled_model_variant_key)
        return exists

    @override
    async def add_model_partition(
        self,
        model_partition: ModelPartition,
    ) -> None:
        set_name = await build_model_partition_dict_name(
            model_partition.model_partition_id
        )
        partition_key = await build_model_partition_key(
            model_partition.model_partition_id
        )

        await self._redis.hset(
            set_name,
            partition_key,
            model_partition.model_dump_json(),
        )
