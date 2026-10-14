from typing import override

from front_end.application.ports.inbound.inference import (
    InferenceRunner,
    LocalWorkerIdentityQuery,
)
from front_end.application.ports.outbound.worker_gateway import (
    WorkerIdentityGateway,
    WorkerIngressGateway,
)
from front_end.domain.inference_result import InferenceResult
from shared.flow.flow import FlowId
from shared.service.service import WorkerId
from shared.tensor.tensor_bundle import TensorBundle


class DefaultInference(InferenceRunner, LocalWorkerIdentityQuery):
    def __init__(
        self,
        worker_ingress_gateway: WorkerIngressGateway,
        worker_identity_gateway: WorkerIdentityGateway,
    ) -> None:
        self._worker_ingress_gateway = worker_ingress_gateway
        self._worker_identity_gateway = worker_identity_gateway

    @override
    async def run_inference(
        self,
        flow_id: FlowId,
        bundle: TensorBundle,
    ) -> InferenceResult:
        return await self._worker_ingress_gateway.run_model_invocation(
            flow_id=flow_id,
            bundle=bundle,
        )

    @override
    async def get_local_worker_identity(self) -> WorkerId:
        return await self._worker_identity_gateway.get_worker_identity()
