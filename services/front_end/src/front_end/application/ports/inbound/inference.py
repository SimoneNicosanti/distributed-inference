from abc import ABC, abstractmethod

from front_end.domain.inference_result import InferenceResult
from shared.flow.flow import FlowId
from shared.service.service import WorkerId
from shared.tensor.tensor_bundle import TensorBundle


class InferenceRunner(ABC):
    @abstractmethod
    async def run_inference(
        self,
        flow_id: FlowId,
        bundle: TensorBundle,
    ) -> InferenceResult: ...


class LocalWorkerIdentityQuery(ABC):
    @abstractmethod
    async def get_local_worker_identity(self) -> WorkerId: ...
