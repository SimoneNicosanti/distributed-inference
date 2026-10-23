from abc import ABC, abstractmethod

from shared.flow.flow import FlowId
from worker.domain.model_invocation.model_invocation import ModelInvocationResult
from worker.domain.tensor.tensor_bundle import TensorBundle


class ModelInvocationCoordinator(ABC):
    @abstractmethod
    async def run_model_invocation(
        self,
        flow_id: FlowId,
        bundle: TensorBundle,
    ) -> ModelInvocationResult: ...
