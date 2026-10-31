from abc import ABC, abstractmethod

from front_end.domain.inference_result import InferenceRequest, InferenceResult
from front_end.domain.model_contract import ModelContract
from shared.flow.flow import FlowId


class InferenceCoordinator(ABC):
    @abstractmethod
    async def get_current_contract_for_flow(self, flow_id: FlowId) -> ModelContract: ...

    @abstractmethod
    async def run_inference_for_request(
        self,
        request: InferenceRequest,
    ) -> InferenceResult: ...
