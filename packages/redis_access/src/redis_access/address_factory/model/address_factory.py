from abc import ABC, abstractmethod
from typing import override

from redis_access.contract.redis_address import KeyRedisAddress, RedisAddress
from shared.model.model import ModelId


class ModelAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    async def build_model_address(model_id: ModelId) -> AddrT: ...


class KeyModelAddressFactory(ModelAddressFactory[KeyRedisAddress]):
    @staticmethod
    @override
    async def build_model_address(model_id: ModelId) -> KeyRedisAddress:
        return KeyRedisAddress(key=f"model:{model_id.model_name}")
