from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model.execution.sub_model_execution_input_output import (
    SubModelExecutionInput,
    SubModelExecutionOutput,
)


## This is the inference worker for a single sub-model
class SubModelExecutor(ABC):
    @abstractmethod
    async def process_sub_model_inference_input(
        self, sub_model_execution_input: SubModelExecutionInput
    ) -> SubModelExecutionOutput: ...

    ## close method can be useful for cleaning up resources
    @abstractmethod
    async def close(self) -> None: ...
