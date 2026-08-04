from pydantic import BaseModel, ConfigDict

from distributed_inference.worker.domain.sub_model.execution.sub_model_execution_context import (
    SubModelExecutionContext,
)
from distributed_inference.worker.domain.tensor.tensor import TensorBundle

## NOTE: To handle stateful models, we will need to add a sort of state in the input/output
## The state should be handled externally, since an executor might be a replicated model; as such, multiple
## replicas can be used to handle the same state


## This is the input for the local sub-model inference
class SubModelExecutionInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_execution_context: SubModelExecutionContext

    payload: TensorBundle


## This is the output of the local sub-model inference
class SubModelExecutionOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_execution_context: SubModelExecutionContext

    payload: TensorBundle
