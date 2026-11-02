from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from integration.dto.context.context_dto import (
    ExternalHopDto,
    HopInvocationContextDto,
    ModelInvocationContextDto,
    ModelPassContextDto,
)
from shared.context.model_pass_context import ModelPassContext, ModelPassType


def model_pass_context_to_dto(context: ModelPassContext) -> ModelPassContextDto:
    if context.model_pass_type != ModelPassType.FORWARD:
        raise ValueError(f"Unsupported model pass type: {context.model_pass_type}")
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


def external_source_context_to_dto(
    context: ExternalSourceInvocationContext,
) -> HopInvocationContextDto:
    return HopInvocationContextDto(
        model_pass_context=model_pass_context_to_dto(context.model_pass_context),
        hop=ExternalHopDto(),
        hop_invocation_id=context.external_src_invocation_id,
    )
