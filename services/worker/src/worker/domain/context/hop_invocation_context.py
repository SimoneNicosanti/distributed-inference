from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shared.context.model_pass_context import ModelPassContext
from shared.plan.hop import PlannedHopId

type HopInvocationId = UUID


## A model pass is made up of multiple hop invocations
## The context of a hop invocation is identified by
## - The model pass it belongs to
## - The invoked hop
## - The hop invocation id
@dataclass(frozen=True, slots=True)
class HopInvocationContext[HopT: PlannedHopId]:
    model_pass_context: ModelPassContext
    hop: HopT
    hop_invocation_id: HopInvocationId = field(default_factory=uuid4)
