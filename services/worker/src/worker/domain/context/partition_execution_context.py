## A partition invocation can lead to multiple partition executions.
## For example in case of fault tolerance.
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from worker.domain.context.partition_invocation_context import (
    PartitionInvocationContext,
)

type PartitionExecutionId = UUID


@dataclass(frozen=True, slots=True)
class PartitionExecutionContext:
    partition_invocation_context: PartitionInvocationContext
    partition_execution_id: PartitionExecutionId = field(default_factory=uuid4)
