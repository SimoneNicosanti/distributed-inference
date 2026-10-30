from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shared.service.service import IngressId
from worker.domain.context.model_pass_context import ModelPassContext

type IngressInvocationId = UUID


@dataclass(frozen=True, slots=True)
class IngressInvocationContext:
    model_pass_context: ModelPassContext
    ingress_id: IngressId
    ingress_invocation_id: IngressInvocationId = field(default_factory=uuid4)
