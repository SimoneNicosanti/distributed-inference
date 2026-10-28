import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import override

from integration.shared.id_json import (
    model_partition_id_to_json,
    model_variant_id_to_json,
)
from shared.model.model_partition import ModelPartitionId


class ModelPartitionKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_model_partition_dict_name(
        model_partition_id: ModelPartitionId,
    ) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_model_partition_key(
        model_partition_id: ModelPartitionId,
    ) -> str: ...


class HashModelPartitionKeyFactory(ModelPartitionKeyFactory):
    @staticmethod
    @override
    async def build_model_partition_dict_name(
        model_partition_id: ModelPartitionId,
    ) -> str:
        model_variant_id_md5 = await asyncio.to_thread(
            hashlib.md5,
            model_variant_id_to_json(model_partition_id.model_variant_id).encode(
                "utf-8"
            ),
        )

        return f"model-partitions:{model_variant_id_md5.hexdigest()}"

    @staticmethod
    @override
    async def build_model_partition_key(model_partition_id: ModelPartitionId) -> str:
        model_partition_id_md5 = await asyncio.to_thread(
            hashlib.md5,
            model_partition_id_to_json(model_partition_id).encode("utf-8"),
        )
        return model_partition_id_md5.hexdigest()


class StringModelPartitionKeyFactory(ModelPartitionKeyFactory):
    @staticmethod
    @override
    async def build_model_partition_dict_name(
        model_partition_id: ModelPartitionId,
    ) -> str:
        full_variant_tag = model_partition_id.model_variant_id.full_variant_tag
        return f"model-partitions:{full_variant_tag}"

    @staticmethod
    @override
    async def build_model_partition_key(model_partition_id: ModelPartitionId) -> str:
        model_partition_id_md5 = await asyncio.to_thread(
            hashlib.md5,
            model_partition_id_to_json(model_partition_id).encode("utf-8"),
        )
        return model_partition_id_md5.hexdigest()
