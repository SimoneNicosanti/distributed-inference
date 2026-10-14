from abc import ABC, abstractmethod

from front_end.domain.inference_result import InferenceResult
from shared.flow.flow import FlowId
from shared.service.service import WorkerId
from shared.tensor.tensor_bundle import TensorBundle


class WorkerIngressGateway(ABC):
    @abstractmethod
    async def run_model_invocation(
        self,
        flow_id: FlowId,
        bundle: TensorBundle,
    ) -> InferenceResult: ...


class WorkerIdentityGateway(ABC):
    @abstractmethod
    async def get_worker_identity(self) -> WorkerId: ...
