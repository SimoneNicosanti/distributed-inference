from integration.dto.context.context_dto import (
    IngressInvocationContextDto,
    ModelInvocationContextDto,
    ModelPassContextDto,
    PartitionInvocationContextDto,
    SourceInvocationContextDto,
)
from shared.context.ingress_invocation_context import IngressInvocationContext
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext
from worker.domain.context.partition_invocation_context import (
    PartitionInvocationContext,
)
from worker.domain.partition.partition_invocation_contribution import (
    SourceInvocationContext,
)


def model_pass_context_to_dto(context: ModelPassContext) -> ModelPassContextDto:
    invocation = context.model_invocation_context
    return ModelPassContextDto(
        model_invocation_context=ModelInvocationContextDto(
            flow_id=invocation.flow_id,
            result_sink=invocation.result_sink,
            model_invocation_id=invocation.model_invocation_id,
        ),
        plan_version=context.plan_version,
        model_pass_type=context.model_pass_type,
        model_pass_id=context.model_pass_id,
    )


def model_pass_context_from_dto(context: ModelPassContextDto) -> ModelPassContext:
    invocation = context.model_invocation_context
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=invocation.flow_id,
            result_sink=invocation.result_sink,
            model_invocation_id=invocation.model_invocation_id,
        ),
        plan_version=context.plan_version,
        model_pass_type=context.model_pass_type,
        model_pass_id=context.model_pass_id,
    )


def source_context_to_dto(
    context: SourceInvocationContext,
) -> SourceInvocationContextDto:
    model_pass_context = model_pass_context_to_dto(context.model_pass_context)
    if isinstance(context, PartitionInvocationContext):
        return PartitionInvocationContextDto(
            model_pass_context=model_pass_context,
            partition_replica_id=context.partition_replica_id,
            partition_invocation_id=context.partition_invocation_id,
        )
    return IngressInvocationContextDto(
        model_pass_context=model_pass_context,
        ingress_invocation_id=context.ingress_invocation_id,
    )


def source_context_from_dto(
    context: SourceInvocationContextDto,
) -> SourceInvocationContext:
    model_pass_context = model_pass_context_from_dto(context.model_pass_context)
    if isinstance(context, PartitionInvocationContextDto):
        return PartitionInvocationContext(
            model_pass_context=model_pass_context,
            partition_replica_id=context.partition_replica_id,
            partition_invocation_id=context.partition_invocation_id,
        )
    return IngressInvocationContext(
        model_pass_context=model_pass_context,
        ingress_invocation_id=context.ingress_invocation_id,
    )
