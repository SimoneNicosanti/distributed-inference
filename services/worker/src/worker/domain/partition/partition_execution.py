from pydantic import BaseModel, ConfigDict

from worker.domain.context.partition_execution_context import (
    PartitionExecutionContext,
)
from worker.domain.partition.tensor_bundle import TensorBundle

## NOTE: To handle stateful models, we will need to add a sort of state in the input/output
## The state should be handled externally, since an executor might be a replicated model; as such, multiple
## replicas can be used to handle the same state


## Input for local partition execution.
class PartitionExecutionInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition_execution_context: PartitionExecutionContext

    payload: TensorBundle


## Output of local partition execution.
class PartitionExecutionOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition_execution_context: PartitionExecutionContext

    payload: TensorBundle
