from abc import ABC, abstractmethod

from front_end.domain.context.model_pass_context import ModelPassContext
from front_end.domain.tensor_map import TensorMap
from shared.plan.hop_deployment_id import HopDeploymentId
from shared.plan.hop_logical_id import SinkReplicaId


class SinkGateway(ABC):
    @abstractmethod
    async def retrieve_result(
        self,
        model_pass_context: ModelPassContext,
        sink_deployment_id: HopDeploymentId[SinkReplicaId],
    ) -> TensorMap: ...
