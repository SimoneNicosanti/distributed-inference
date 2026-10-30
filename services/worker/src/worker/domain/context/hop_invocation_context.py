from dataclasses import dataclass, field
from uuid import UUID, uuid4

from worker.domain.context.model_pass_context import ModelPassContext
from worker.domain.hop.hop import Hop

type HopInvocationId = UUID


## A model pass is made up of multiple hop invocations
## The context of a hop invocation is identified by
## - The model pass it belongs to
## - The invoked hop
## - The hop invocation id
@dataclass(frozen=True, slots=True)
class HopInvocationContext[HopT: Hop]:
    model_pass_context: ModelPassContext
    hop: HopT
    hop_invocation_id: HopInvocationId = field(default_factory=uuid4)
