from abc import ABC, abstractmethod

from front_end.domain.inference_result import InferenceResult
from front_end.domain.tensor_map import TensorMap
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class WorkerIngressGateway(ABC):
    @abstractmethod
    async def run_model_invocation(
        self,
        flow_id: FlowId,
        bundle: TensorMap,
    ) -> InferenceResult: ...


class WorkerIdentityGateway(ABC):
    @abstractmethod
    async def get_worker_identity(self) -> WorkerId: ...
