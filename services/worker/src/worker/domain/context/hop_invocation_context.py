from dataclasses import dataclass, field
from uuid import UUID, uuid4

from worker.domain.context.model_pass_context import ModelPassContext
from worker.domain.plan.hop.hop import Hop

type HopInvocationId = UUID


@dataclass(frozen=True, slots=True)
class HopInvocationContext:
    model_pass_context: ModelPassContext
    hop: Hop
    hop_invocation_id: HopInvocationId = field(default_factory=uuid4)
