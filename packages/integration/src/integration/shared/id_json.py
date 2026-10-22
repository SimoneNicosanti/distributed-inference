from pydantic import TypeAdapter

from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.partition_replica_id import PartitionReplicaId

_FLOW_ID_ADAPTER = TypeAdapter(FlowId)
_MODEL_ID_ADAPTER = TypeAdapter(ModelId)
_MODEL_VARIANT_ID_ADAPTER = TypeAdapter(ModelVariantId)
_MODEL_PARTITION_ID_ADAPTER = TypeAdapter(ModelPartitionId)
_PARTITION_REPLICA_ID_ADAPTER = TypeAdapter(PartitionReplicaId)
_PARTITION_DEPLOYMENT_ID_ADAPTER = TypeAdapter(PartitionDeploymentId)


def flow_id_to_json(value: FlowId) -> str:
    return _FLOW_ID_ADAPTER.dump_json(value).decode()


def flow_id_from_json(value: str | bytes) -> FlowId:
    return _FLOW_ID_ADAPTER.validate_json(value)


def model_id_to_json(value: ModelId) -> str:
    return _MODEL_ID_ADAPTER.dump_json(value).decode()


def model_variant_id_to_json(value: ModelVariantId) -> str:
    return _MODEL_VARIANT_ID_ADAPTER.dump_json(value).decode()


def model_variant_id_from_json(value: str | bytes) -> ModelVariantId:
    return _MODEL_VARIANT_ID_ADAPTER.validate_json(value)


def model_partition_id_to_json(value: ModelPartitionId) -> str:
    return _MODEL_PARTITION_ID_ADAPTER.dump_json(value).decode()


def partition_replica_id_to_json(value: PartitionReplicaId) -> str:
    return _PARTITION_REPLICA_ID_ADAPTER.dump_json(value).decode()


def partition_deployment_id_to_json(value: PartitionDeploymentId) -> str:
    return _PARTITION_DEPLOYMENT_ID_ADAPTER.dump_json(value).decode()


def partition_deployment_id_from_json(
    value: str | bytes,
) -> PartitionDeploymentId:
    return _PARTITION_DEPLOYMENT_ID_ADAPTER.validate_json(value)
