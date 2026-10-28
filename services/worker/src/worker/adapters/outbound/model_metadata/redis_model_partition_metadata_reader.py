from typing import override

from redis import asyncio as redis_asyncio

from integration.dto.model.model_partition_dto import ModelPartitionDto
from integration.redis.model_partition.keys_factory import (
    ModelPartitionKeyFactory,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId
from worker.application.ports.outbound.model_metadata.model_partition_metadata_reader import (
    ModelPartitionMetadataReader,
)


class RedisModelPartitionMetadataReader(ModelPartitionMetadataReader):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        model_partition_key_factory_cls: type[ModelPartitionKeyFactory],
    ) -> None:
        self._redis = redis
        self._model_partition_key_factory_cls = model_partition_key_factory_cls

    @override
    async def get_artifact_ref_by_partition_id(
        self, partition_id: ModelPartitionId
    ) -> ArtifactRef:
        dict_name = (
            await self._model_partition_key_factory_cls.build_model_partition_dict_name(
                partition_id
            )
        )
        key = await self._model_partition_key_factory_cls.build_model_partition_key(
            partition_id
        )

        value = await self._redis.hget(dict_name, key)
        if value is None:
            raise KeyError(f"Model partition {partition_id} not found")

        metadata = ModelPartitionDto.model_validate_json(value)
        if metadata.model_partition_id != partition_id:
            raise ValueError("Model partition metadata ID mismatch")

        return metadata.artifact_ref
