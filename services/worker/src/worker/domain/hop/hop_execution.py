from dataclasses import dataclass

from worker.domain.context.hop_execution_context import (
    HopExecutionContext,
)
from worker.domain.hop.hop import ExecutableHop
from worker.domain.tensor.tensor_bundle import TensorBundle

## NOTE: To handle stateful models, we will need to add a sort of state in the input/output
## The state should be handled externally, since an executor might be a replicated model; as such, multiple
## replicas can be used to handle the same state


## Input for local partition execution.
@dataclass(frozen=True, slots=True)
class HopExecutionInput[HopT: ExecutableHop]:
    context: HopExecutionContext[HopT]

    payload: TensorBundle


## Output of local partition execution.
@dataclass(frozen=True, slots=True)
class HopExecutionOutput[HopT: ExecutableHop]:
    context: HopExecutionContext[HopT]

    payload: TensorBundle
