from abc import ABC, abstractmethod
from collections.abc import Iterable

from shared.model.model_partition import ModelPartitionId


class PartitionBuilderGateway(ABC):
    @abstractmethod
    async def build_all_partitions_by_ids(
        self, partition_ids: Iterable[ModelPartitionId]
    ) -> None: ...
