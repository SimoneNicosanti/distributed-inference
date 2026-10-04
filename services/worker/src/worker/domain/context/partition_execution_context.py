## A partition invocation can lead to multiple partition executions.
## For example in case of fault tolerance.
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from worker.domain.context.partition_invocation_context import (
    PartitionInvocationContext,
)

type PartitionExecutionId = UUID


class PartitionExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    partition_invocation_context: PartitionInvocationContext
    partition_execution_id: PartitionExecutionId = Field(default_factory=uuid4)
