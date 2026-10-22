from integration.dto.model.model_partition_dto import ModelPartitionDto
from model_manager.domain.model_partition import ModelPartition


class ModelPartitionMapper:
    @staticmethod
    def to_model_partition_dto(model_partition: ModelPartition) -> ModelPartitionDto:
        return ModelPartitionDto(
            model_partition_id=model_partition.model_partition_id,
            artifact_ref=model_partition.artifact_ref,
        )
