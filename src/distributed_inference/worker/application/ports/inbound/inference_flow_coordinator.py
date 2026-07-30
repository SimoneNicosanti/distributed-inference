from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model_inference_message import (
    SubModelInferenceMessage,
)


## This is the coordinator for the entire inference flow
## 1. Receive inference message
## 2. Call message gather
## 3. If InferenceRequest is ready, then continue
## 5. Call Inference manager
## 6. Wait for response to arrive
## 7. Call result routing
## 8. Forward result
class InferenceFlowCoordinator(ABC):
    @abstractmethod
    async def process_sub_model_inference_message(
        self, sub_model_inference_message: SubModelInferenceMessage
    ) -> None: ...
