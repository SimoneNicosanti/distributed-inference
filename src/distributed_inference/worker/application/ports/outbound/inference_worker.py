from abc import ABC, abstractmethod

from distributed_inference.worker.domain.inference_run import (
    InferenceInput,
    InferenceOutput,
)


## This is the inference worker for a single sub-model
class InferenceWorker(ABC):
    @abstractmethod
    async def process_inference_input(
        self, inference_input: InferenceInput
    ) -> InferenceOutput: ...

    ## close method can be useful for cleaning up resources
    @abstractmethod
    async def close(self) -> None: ...
