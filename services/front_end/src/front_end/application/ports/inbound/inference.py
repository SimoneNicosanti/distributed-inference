from abc import ABC, abstractmethod

from front_end.domain.inference_result import InferenceResult
from front_end.domain.tensor_map import TensorMap
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class InferenceRunner(ABC):
    @abstractmethod
    async def run_inference(
        self,
        flow_id: FlowId,
        bundle: TensorMap,
    ) -> InferenceResult: ...


class LocalWorkerIdentityQuery(ABC):
    @abstractmethod
    async def get_local_worker_identity(self) -> WorkerId: ...
