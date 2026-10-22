from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shared.context.model_pass_context import ModelPassContext

type IngressInvocationId = UUID


@dataclass(frozen=True, slots=True)
class IngressInvocationContext:
    model_pass_context: ModelPassContext
    ingress_invocation_id: IngressInvocationId = field(default_factory=uuid4)
