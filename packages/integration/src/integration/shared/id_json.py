from typing import Annotated, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_deployment_id import HopDeploymentId, HopReplicaId
from shared.plan.hop_logical_id import (
    IngressReplicaId,
    PartitionReplicaId,
    SinkReplicaId,
)
from shared.service.service import ServiceId, WorkerId


class _PartitionReplicaIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["partition"] = "partition"
    value: PartitionReplicaId


class _IngressReplicaIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["ingress"] = "ingress"
    value: IngressReplicaId


class _SinkReplicaIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["sink"] = "sink"
    value: SinkReplicaId


type _HopReplicaIdEnvelope = Annotated[
    _PartitionReplicaIdEnvelope | _IngressReplicaIdEnvelope | _SinkReplicaIdEnvelope,
    Field(discriminator="kind"),
]


class _HopDeploymentIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_id: _HopReplicaIdEnvelope
    worker_id: WorkerId


_FLOW_ID_ADAPTER = TypeAdapter(FlowId)
_MODEL_ID_ADAPTER = TypeAdapter(ModelId)
_MODEL_VARIANT_ID_ADAPTER = TypeAdapter(ModelVariantId)
_MODEL_PARTITION_ID_ADAPTER = TypeAdapter(ModelPartitionId)
_PARTITION_REPLICA_ID_ADAPTER = TypeAdapter(PartitionReplicaId)
_HOP_REPLICA_ID_ADAPTER = TypeAdapter(_HopReplicaIdEnvelope)
_HOP_DEPLOYMENT_ID_ADAPTER = TypeAdapter(_HopDeploymentIdEnvelope)


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


def hop_replica_id_to_json(value: HopReplicaId) -> str:
    return _HOP_REPLICA_ID_ADAPTER.dump_json(
        _hop_replica_id_to_envelope(value)
    ).decode()


def hop_replica_id_from_json(value: str | bytes) -> HopReplicaId:
    envelope = _HOP_REPLICA_ID_ADAPTER.validate_json(value)
    return envelope.value


def hop_deployment_id_to_json[ReplicaIdT: HopReplicaId](
    value: HopDeploymentId[ReplicaIdT],
) -> str:
    envelope = _HopDeploymentIdEnvelope(
        replica_id=_hop_replica_id_to_envelope(value.replica_id),
        worker_id=value.worker_id,
    )
    return _HOP_DEPLOYMENT_ID_ADAPTER.dump_json(envelope).decode()


def hop_deployment_id_from_json(
    value: str | bytes,
) -> HopDeploymentId[HopReplicaId]:
    envelope = _HOP_DEPLOYMENT_ID_ADAPTER.validate_json(value)
    return HopDeploymentId(
        replica_id=envelope.replica_id.value,
        worker_id=_canonical_worker_id(envelope.worker_id),
    )


def partition_deployment_id_to_json(
    value: HopDeploymentId[PartitionReplicaId],
) -> str:
    return hop_deployment_id_to_json(value)


def partition_deployment_id_from_json(
    value: str | bytes,
) -> HopDeploymentId[PartitionReplicaId]:
    deployment_id = hop_deployment_id_from_json(value)
    if not isinstance(deployment_id.replica_id, PartitionReplicaId):
        raise ValueError("Expected a partition hop deployment ID")
    return cast(HopDeploymentId[PartitionReplicaId], deployment_id)


def _hop_replica_id_to_envelope(
    value: HopReplicaId,
) -> _HopReplicaIdEnvelope:
    match value:
        case PartitionReplicaId():
            return _PartitionReplicaIdEnvelope(value=value)
        case IngressReplicaId():
            return _IngressReplicaIdEnvelope(value=value)
        case SinkReplicaId():
            return _SinkReplicaIdEnvelope(value=value)


def _canonical_worker_id(worker_id: WorkerId) -> WorkerId:
    if not isinstance(worker_id.id, str):
        return worker_id

    try:
        return ServiceId(id=UUID(worker_id.id))
    except ValueError:
        return worker_id
