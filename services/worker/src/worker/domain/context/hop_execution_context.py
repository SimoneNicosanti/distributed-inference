## A partition invocation can lead to multiple partition executions.
## For example in case of fault tolerance.
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shared.plan.hop import PlannedExecutableHopId
from worker.domain.context.hop_invocation_context import HopInvocationContext

type HopExecutionId = UUID


## A hop execution can be retried multiple times
## Each execution retry is identified by:
## - hop invocation context
## - hop execution retry id
@dataclass(frozen=True, slots=True)
class HopExecutionContext[HopT: PlannedExecutableHopId]:
    hop_invocation_context: HopInvocationContext[HopT]
    hop_execution_id: HopExecutionId = field(default_factory=uuid4)
