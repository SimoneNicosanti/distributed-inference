from pydantic import BaseModel, ConfigDict

from distributed_inference.worker.domain.sub_model.execution.sub_model_execution_context import (
    SubModelExecutionContext,
)
from distributed_inference.worker.domain.sub_model.execution.sub_model_execution_state import (
    SubModelExecutionState,
)
from distributed_inference.worker.domain.tensor.tensor import TensorBundle


## This is the input for the local sub-model inference
class SubModelExecutionInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_execution_context: SubModelExecutionContext

    payload: TensorBundle
    state: SubModelExecutionState | None


## This is the output of the local sub-model inference
class SubModelExecutionOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_execution_context: SubModelExecutionContext

    payload: TensorBundle
    state: SubModelExecutionState | None
