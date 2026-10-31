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
from shared.plan.hop_logical_id import (
    IngressReplicaId,
    PartitionReplicaId,
    SinkReplicaId,
)
from worker.domain.context.hop_invocation_context import HopInvocationContext
from worker.domain.context.model_invocation_context import ModelInvocationContext
from worker.domain.context.model_pass_context import ModelPassContext, ModelPassType
from worker.domain.hop.hop import (
    DestinationHop,
    ExternalHop,
    Hop,
    InvocableHop,
    SourceHop,
)


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


def hop_to_dto(hop: Hop) -> ExternalHopDto | InvocableHopDto:
    match hop:
        case ExternalHop():
            return ExternalHopDto()
        case IngressReplicaId():
            return IngressHopDto(ingress_replica_id=hop)
        case PartitionReplicaId():
            return PartitionReplicaHopDto(partition_replica_id=hop)
        case SinkReplicaId():
            return SinkHopDto(sink_replica_id=hop)
        case _:
            raise TypeError(f"Unsupported hop: {hop!r}")


def source_hop_from_dto(hop: SourceHopDto) -> SourceHop:
    match hop:
        case ExternalHopDto():
            return ExternalHop()
        case IngressHopDto(ingress_replica_id=ingress_replica_id):
            return ingress_replica_id
        case PartitionReplicaHopDto(partition_replica_id=replica_id):
            return replica_id
        case _:
            raise TypeError(f"Unsupported source hop DTO: {hop!r}")


def destination_hop_from_dto(hop: DestinationHopDto) -> DestinationHop:
    match hop:
        case IngressHopDto(ingress_replica_id=ingress_replica_id):
            return ingress_replica_id
        case PartitionReplicaHopDto(partition_replica_id=replica_id):
            return replica_id
        case SinkHopDto(sink_replica_id=sink_replica_id):
            return sink_replica_id
        case _:
            raise TypeError(f"Unsupported destination hop DTO: {hop!r}")


def destination_hop_to_dto(hop: DestinationHop) -> DestinationHopDto:
    dto = hop_to_dto(hop)
    if not isinstance(dto, (IngressHopDto, PartitionReplicaHopDto, SinkHopDto)):
        raise TypeError(f"Unsupported destination hop: {hop!r}")
    return dto


def invocable_hop_from_dto(hop: InvocableHopDto) -> InvocableHop:
    return destination_hop_from_dto(hop)


def hop_invocation_context_to_dto(
    context: HopInvocationContext[SourceHop],
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
) -> HopInvocationContext[SourceHop]:
    return HopInvocationContext(
        model_pass_context=model_pass_context_from_dto(context.model_pass_context),
        hop=source_hop_from_dto(context.hop),
        hop_invocation_id=context.hop_invocation_id,
    )
