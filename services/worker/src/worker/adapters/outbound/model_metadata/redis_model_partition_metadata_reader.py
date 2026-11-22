from typing import override

from redis import asyncio as redis_asyncio

from integration.dto.model.model_partition_dto import ModelPartitionDto
from redis_access.address_factory.model_partition.address_factory import (
    ModelPartitionAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.redis_reader.abc.single_value_redis_reader import (
    RedisReadError,
    SingleValueRedisReader,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId
from worker.application.ports.outbound.model_metadata.model_partition_metadata_reader import (
    ArtifactReferenceReadError,
    ModelPartitionMetadataReader,
)


class RedisModelPartitionMetadataReader(ModelPartitionMetadataReader):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        redis_reader: SingleValueRedisReader[HashMapRedisAddress],
        address_factory: ModelPartitionAddressFactory[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        self._redis_reader = redis_reader
        self._address_factory = address_factory

    @override
    async def get_artifact_ref_by_partition_id(
        self, partition_id: ModelPartitionId
    ) -> ArtifactRef:
        try:
            addr = await self._address_factory.build_model_partition_address(
                partition_id
            )

            dto_json = await self._redis_reader.read_single(self._redis, addr)

            dto = ModelPartitionDto.model_validate_json(dto_json)
            if dto.model_partition_id != partition_id:
                raise ValueError("Model partition metadata ID mismatch")

            return dto.artifact_ref
        except (ValueError, RedisReadError) as e:
            raise ArtifactReferenceReadError("Could not read artifact reference") from e
