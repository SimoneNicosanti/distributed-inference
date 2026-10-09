from typing import override

from pydantic import BaseModel, ConfigDict
from redis import asyncio as redis_asyncio

from integration.redis.model_partition.keys_factory import (
    build_model_partition_dict_name,
    build_model_partition_key,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId
from worker.application.ports.outbound.model_metadata.model_partition_metadata_reader import (
    ModelPartitionMetadataReader,
)


class _ModelPartitionMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_partition_id: ModelPartitionId
    artifact_ref: ArtifactRef


class RedisModelPartitionMetadataReader(ModelPartitionMetadataReader):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_artifact_ref_by_partition_id(
        self, partition_id: ModelPartitionId
    ) -> ArtifactRef:
        dict_name = await build_model_partition_dict_name(partition_id)
        key = await build_model_partition_key(partition_id)

        value = await self._redis.hget(dict_name, key)
        if value is None:
            raise KeyError(f"Model partition {partition_id} not found")

        metadata = _ModelPartitionMetadata.model_validate_json(value)
        if metadata.model_partition_id != partition_id:
            raise ValueError("Model partition metadata ID mismatch")

        return metadata.artifact_ref
