import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    RedisAddress,
)
from shared.model.model_partition import ModelPartitionId


class ModelPartitionAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    async def build_model_partition_address(
        model_partition_id: ModelPartitionId,
    ) -> AddrT: ...


class HashMapModelPartitionAddressFactory(
    ModelPartitionAddressFactory[HashMapRedisAddress]
):
    @staticmethod
    @override
    async def build_model_partition_address(
        model_partition_id: ModelPartitionId,
    ) -> HashMapRedisAddress:
        full_variant_tag = model_partition_id.model_variant_id.full_variant_tag
        model_partition_id_md5 = await asyncio.to_thread(
            hashlib.md5,
            json.dumps(
                asdict(model_partition_id), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
        )
        return HashMapRedisAddress(
            hash_map_key=f"model-partitions:{full_variant_tag}",
            field_key=model_partition_id_md5.hexdigest(),
        )
