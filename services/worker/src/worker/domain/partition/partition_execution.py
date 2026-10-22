from dataclasses import dataclass

from shared.tensor.tensor_bundle import TensorBundle
from worker.domain.context.partition_execution_context import (
    PartitionExecutionContext,
)

## NOTE: To handle stateful models, we will need to add a sort of state in the input/output
## The state should be handled externally, since an executor might be a replicated model; as such, multiple
## replicas can be used to handle the same state


## Input for local partition execution.
@dataclass(frozen=True, slots=True)
class PartitionExecutionInput:
    partition_execution_context: PartitionExecutionContext

    payload: TensorBundle


## Output of local partition execution.
@dataclass(frozen=True, slots=True)
class PartitionExecutionOutput:
    partition_execution_context: PartitionExecutionContext

    payload: TensorBundle
