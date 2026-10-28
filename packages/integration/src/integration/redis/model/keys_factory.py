import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import override

from integration.shared.id_json import model_id_to_json
from shared.model.model import ModelId


class ModelKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_model_key(model_id: ModelId) -> str: ...


class HashModelKeyFactory(ModelKeyFactory):
    @staticmethod
    @override
    async def build_model_key(model_id: ModelId) -> str:
        md5_value = await asyncio.to_thread(
            hashlib.md5, model_id_to_json(model_id).encode("utf-8")
        )
        return f"model:{md5_value.hexdigest()}"


class StringModelKeyFactory(ModelKeyFactory):
    @staticmethod
    @override
    async def build_model_key(model_id: ModelId) -> str:
        return f"model:{model_id.model_name}"
