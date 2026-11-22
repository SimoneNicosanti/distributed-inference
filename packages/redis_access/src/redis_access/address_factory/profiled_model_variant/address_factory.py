from abc import ABC, abstractmethod
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    KeyRedisAddress,
    RedisAddress,
)
from shared.model.model_variant import ModelVariantId


class ProfiledModelVariantAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    async def build_profiled_model_variant_address(
        model_variant_id: ModelVariantId,
    ) -> AddrT: ...

    @staticmethod
    def build_profiled_model_variant_ids_address() -> KeyRedisAddress:
        return KeyRedisAddress(key="profiled-model-variant:ids")


class HashMapProfiledModelVariantAddressFactory(
    ProfiledModelVariantAddressFactory[HashMapRedisAddress]
):
    @staticmethod
    @override
    async def build_profiled_model_variant_address(
        model_variant_id: ModelVariantId,
    ) -> HashMapRedisAddress:
        return HashMapRedisAddress(
            hash_map_key=f"profiled-model-variant:{model_variant_id.model_id.model_name}",
            field_key=model_variant_id.variant_tag,
        )
