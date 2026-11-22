from abc import ABC, abstractmethod
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    RedisAddress,
)
from shared.service.service import WorkerId


class ResourceProfileAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    def build_worker_resource_profile_address(worker_id: WorkerId) -> AddrT: ...


class HashMapResourceProfileAddressFactory(
    ResourceProfileAddressFactory[HashMapRedisAddress]
):
    @staticmethod
    @override
    def build_worker_resource_profile_address(
        worker_id: WorkerId,
    ) -> HashMapRedisAddress:
        return HashMapRedisAddress(
            hash_map_key="worker-profiles:resource", field_key=f"{worker_id.id}"
        )
