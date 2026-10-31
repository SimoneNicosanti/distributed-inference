from dataclasses import dataclass, field
from uuid import UUID, uuid4

from front_end.domain.context.model_pass_context import ModelPassContext

type ExternalSourceInvocationId = UUID


@dataclass(frozen=True, slots=True)
class ExternalSourceInvocationContext:
    model_pass_context: ModelPassContext
    external_src_invocation_id: ExternalSourceInvocationId = field(
        default_factory=uuid4
    )
