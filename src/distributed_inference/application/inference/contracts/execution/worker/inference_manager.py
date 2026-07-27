from abc import ABC, abstractmethod

from distributed_inference.domain.identifiers import SubModelId


## This is the inference manager for a pool of sub-models
class InferenceManager(ABC):
    @abstractmethod
    def load_sub_model(self, sub_model_id: SubModelId) -> None: ...

    @abstractmethod
    def unload_sub_model(self, sub_model_id: SubModelId) -> None: ...

    @abstractmethod
    def run_inference_request(self, request: InferenceRequest) -> InferenceResponse: ...
