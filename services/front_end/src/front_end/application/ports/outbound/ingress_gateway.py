from abc import ABC, abstractmethod

from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.tensor_map import TensorMap
from shared.plan.hop_deployment_id import HopDeploymentId
from shared.plan.hop_logical_id import IngressReplicaId


class IngressGateway(ABC):
    @abstractmethod
    async def submit_model_pass(
        self,
        external_source_invocation_context: ExternalSourceInvocationContext,
        bundle: TensorMap,
        ingress_deployment_id: HopDeploymentId[IngressReplicaId],
    ) -> None: ...
