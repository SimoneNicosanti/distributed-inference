from integration.dto.context.context_dto import (
    DestinationHopDto,
    ExternalHopDto,
    HopInvocationContextDto,
    IngressHopDto,
    InvocableHopDto,
    ModelInvocationContextDto,
    ModelPassContextDto,
    PartitionReplicaHopDto,
    SinkHopDto,
    SourceHopDto,
)
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.plan.hop import (
    ExternalHopId,
    PlannedDestinationHopId,
    PlannedHopId,
    PlannedInvocableHopId,
    PlannedSourceHopId,
)
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from worker.domain.context.hop_invocation_context import HopInvocationContext


def model_pass_context_to_dto(context: ModelPassContext) -> ModelPassContextDto:
    invocation = context.model_invocation_context
    return ModelPassContextDto(
        model_invocation_context=ModelInvocationContextDto(
            flow_id=invocation.flow_id,
            model_invocation_id=invocation.model_invocation_id,
        ),
        plan_version=context.plan_version,
        model_pass_type="forward",
        model_pass_id=context.model_pass_id,
    )


def model_pass_context_from_dto(context: ModelPassContextDto) -> ModelPassContext:
    invocation = context.model_invocation_context
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=invocation.flow_id,
            model_invocation_id=invocation.model_invocation_id,
        ),
        plan_version=context.plan_version,
        model_pass_type=ModelPassType(context.model_pass_type),
        model_pass_id=context.model_pass_id,
    )


def hop_to_dto(hop: PlannedHopId) -> ExternalHopDto | InvocableHopDto:
    match hop:
        case ExternalHopId():
            return ExternalHopDto()
        case PlannedIngressReplicaId():
            return IngressHopDto(ingress_replica_id=hop)
        case PlannedPartitionReplicaId():
            return PartitionReplicaHopDto(partition_replica_id=hop)
        case PlannedSinkReplicaId():
            return SinkHopDto(sink_replica_id=hop)
        case _:
            raise TypeError(f"Unsupported hop: {hop!r}")


def source_hop_from_dto(hop: SourceHopDto) -> PlannedSourceHopId:
    match hop:
        case ExternalHopDto():
            return ExternalHopId()
        case IngressHopDto(ingress_replica_id=ingress_replica_id):
            return ingress_replica_id
        case PartitionReplicaHopDto(partition_replica_id=replica_id):
            return replica_id
        case _:
            raise TypeError(f"Unsupported source hop DTO: {hop!r}")


def destination_hop_from_dto(hop: DestinationHopDto) -> PlannedDestinationHopId:
    match hop:
        case IngressHopDto(ingress_replica_id=ingress_replica_id):
            return ingress_replica_id
        case PartitionReplicaHopDto(partition_replica_id=replica_id):
            return replica_id
        case SinkHopDto(sink_replica_id=sink_replica_id):
            return sink_replica_id
        case _:
            raise TypeError(f"Unsupported destination hop DTO: {hop!r}")


def destination_hop_to_dto(hop: PlannedDestinationHopId) -> DestinationHopDto:
    dto = hop_to_dto(hop)
    if not isinstance(dto, (IngressHopDto, PartitionReplicaHopDto, SinkHopDto)):
        raise TypeError(f"Unsupported destination hop: {hop!r}")
    return dto


def invocable_hop_from_dto(hop: InvocableHopDto) -> PlannedInvocableHopId:
    return destination_hop_from_dto(hop)


def hop_invocation_context_to_dto(
    context: HopInvocationContext[PlannedSourceHopId],
) -> HopInvocationContextDto:
    hop = hop_to_dto(context.hop)
    if not isinstance(hop, (ExternalHopDto, IngressHopDto, PartitionReplicaHopDto)):
        raise TypeError(f"Unsupported source hop: {context.hop!r}")
    return HopInvocationContextDto(
        model_pass_context=model_pass_context_to_dto(context.model_pass_context),
        hop=hop,
        hop_invocation_id=context.hop_invocation_id,
    )


def hop_invocation_context_from_dto(
    context: HopInvocationContextDto,
) -> HopInvocationContext[PlannedSourceHopId]:
    return HopInvocationContext(
        model_pass_context=model_pass_context_from_dto(context.model_pass_context),
        hop=source_hop_from_dto(context.hop),
        hop_invocation_id=context.hop_invocation_id,
    )
