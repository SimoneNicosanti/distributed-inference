from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model_inference_input_output import (
    SubModelInferenceInput,
    SubModelInferenceOutput,
)


## This is the inference worker for a single sub-model
class SubModelExecutor(ABC):
    @abstractmethod
    async def process_sub_model_inference_input(
        self, sub_model_inference_input: SubModelInferenceInput
    ) -> SubModelInferenceOutput: ...

    ## close method can be useful for cleaning up resources
    @abstractmethod
    async def close(self) -> None: ...
