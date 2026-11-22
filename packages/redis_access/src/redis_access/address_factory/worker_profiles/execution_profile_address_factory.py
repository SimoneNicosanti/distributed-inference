import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    KeyRedisAddress,
    RedisAddress,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class ExecutionProfileAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    async def build_worker_model_execution_profile_address(
        worker_id: WorkerId, model_variant_id: ModelVariantId
    ) -> AddrT: ...

    @staticmethod
    @abstractmethod
    async def build_worker_execution_profiles_address(
        worker_id: WorkerId,
    ) -> KeyRedisAddress: ...


class HashMapExecutionProfileAddressFactory(
    ExecutionProfileAddressFactory[HashMapRedisAddress]
):
    @staticmethod
    @override
    async def build_worker_model_execution_profile_address(
        worker_id: WorkerId, model_variant_id: ModelVariantId
    ) -> HashMapRedisAddress:

        container = await HashMapExecutionProfileAddressFactory.build_worker_execution_profiles_address(
            worker_id
        )
        return HashMapRedisAddress(
            hash_map_key=container.key,
            field_key=json.dumps(
                asdict(model_variant_id),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    @staticmethod
    @override
    async def build_worker_execution_profiles_address(
        worker_id: WorkerId,
    ) -> KeyRedisAddress:
        return KeyRedisAddress(key=f"worker-profiles:execution:{worker_id.id}")
