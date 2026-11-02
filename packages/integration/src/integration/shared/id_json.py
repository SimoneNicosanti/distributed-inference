from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop import PlannedInvocableHopId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from shared.service.service import ServiceId, WorkerId


class _PartitionReplicaIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["partition"] = "partition"
    value: PlannedPartitionReplicaId


class _IngressReplicaIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["ingress"] = "ingress"
    value: PlannedIngressReplicaId


class _SinkReplicaIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["sink"] = "sink"
    value: PlannedSinkReplicaId


type _HopReplicaIdEnvelope = Annotated[
    _PartitionReplicaIdEnvelope | _IngressReplicaIdEnvelope | _SinkReplicaIdEnvelope,
    Field(discriminator="kind"),
]


class _HopDeploymentIdEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_id: _HopReplicaIdEnvelope
    worker_id: WorkerId


_SERVICE_ID_ADAPTER = TypeAdapter(ServiceId)
_FLOW_ID_ADAPTER = TypeAdapter(FlowId)
_MODEL_ID_ADAPTER = TypeAdapter(ModelId)
_MODEL_VARIANT_ID_ADAPTER = TypeAdapter(ModelVariantId)
_MODEL_PARTITION_ID_ADAPTER = TypeAdapter(ModelPartitionId)
_PARTITION_REPLICA_ID_ADAPTER = TypeAdapter(PlannedPartitionReplicaId)
_HOP_REPLICA_ID_ADAPTER = TypeAdapter(_HopReplicaIdEnvelope)
_HOP_DEPLOYMENT_ID_ADAPTER = TypeAdapter(_HopDeploymentIdEnvelope)


def service_id_to_json(value: ServiceId) -> str:
    return _SERVICE_ID_ADAPTER.dump_json(value).decode()


def service_id_from_json(value: str | bytes) -> ServiceId:
    return _canonical_worker_id(_SERVICE_ID_ADAPTER.validate_json(value))


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


def partition_replica_id_to_json(value: PlannedPartitionReplicaId) -> str:
    return _PARTITION_REPLICA_ID_ADAPTER.dump_json(value).decode()


def planned_invocable_hop_id_to_json(value: PlannedInvocableHopId) -> str:
    return _HOP_REPLICA_ID_ADAPTER.dump_json(
        _planned_invocable_hop_id_to_envelope(value)
    ).decode()


def planned_invocable_hop_id_from_json(value: str | bytes) -> PlannedInvocableHopId:
    envelope = _HOP_REPLICA_ID_ADAPTER.validate_json(value)
    return envelope.value


def _planned_invocable_hop_id_to_envelope(
    value: PlannedInvocableHopId,
) -> _HopReplicaIdEnvelope:
    match value:
        case PlannedPartitionReplicaId():
            return _PartitionReplicaIdEnvelope(value=value)
        case PlannedIngressReplicaId():
            return _IngressReplicaIdEnvelope(value=value)
        case PlannedSinkReplicaId():
            return _SinkReplicaIdEnvelope(value=value)


def _canonical_worker_id(worker_id: WorkerId) -> WorkerId:
    if not isinstance(worker_id.id, str):
        return worker_id

    try:
        return ServiceId(id=UUID(worker_id.id))
    except ValueError:
        return worker_id
