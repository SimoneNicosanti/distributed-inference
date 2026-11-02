from abc import ABC, abstractmethod

from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.tensor_map import TensorMap
from shared.plan.hop_logical_id import PlannedIngressReplicaId
from shared.service.service import WorkerId


class IngressGateway(ABC):
    @abstractmethod
    async def submit_model_pass(
        self,
        external_source_invocation_context: ExternalSourceInvocationContext,
        bundle: TensorMap,
        planned_ingress_replica_id: PlannedIngressReplicaId,
        worker_id: WorkerId,
    ) -> None: ...
