from abc import ABC, abstractmethod
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    KeyRedisAddress,
    RedisAddress,
)
from shared.service.service import ServiceId


class ServiceInstanceAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    def build_service_instance_address(service_id: ServiceId) -> AddrT: ...

    @staticmethod
    def build_service_catalog_address() -> KeyRedisAddress:
        return KeyRedisAddress(key="catalog:services")


class HashMapServiceInstanceAddressFactory(
    ServiceInstanceAddressFactory[HashMapRedisAddress]
):
    @staticmethod
    @override
    def build_service_instance_address(service_id: ServiceId) -> HashMapRedisAddress:
        return HashMapRedisAddress(
            hash_map_key="catalog:services", field_key=f"{service_id.id}"
        )
