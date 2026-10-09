from abc import ABC, abstractmethod

from ingress.domain.model_invocation_result import ModelInvocationResult
from shared.flow.flow import FlowId
from shared.tensor.tensor_bundle import TensorBundle


class ModelInvocationCoordinator(ABC):
    @abstractmethod
    async def run_model_invocation(
        self,
        flow_id: FlowId,
        bundle: TensorBundle,
    ) -> ModelInvocationResult: ...
