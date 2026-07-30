## A sub-model invocation can lead to multiple sub-models executions
## For example in case of fault tolerance.
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationContext,
)

type SubModelExecutionId = UUID


class SubModelExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    sub_model_invocation_context: SubModelInvocationContext
    sub_model_execution_id: SubModelExecutionId = Field(default_factory=uuid4)
