from typing import override

from front_end.application.ports.inbound.inference_runner import (
    InferenceRunner,
    LocalWorkerIdentityQuery,
)
from front_end.application.ports.outbound.front_end_plan_reader import (
    FrontEndPlanReader,
)
from front_end.application.ports.outbound.worker_gateway import (
    WorkerIngressGateway,
)
from front_end.domain.inference_result import InferenceRequest, InferenceResult
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class DefaultInferenceRunner(InferenceRunner, LocalWorkerIdentityQuery):
    def __init__(
        self,
        front_end_plan_reader: FrontEndPlanReader,
        worker_ingress_gateway: WorkerIngressGateway,
    ) -> None:
        self._front_end_plan_reader = front_end_plan_reader
        self._worker_ingress_gateway = worker_ingress_gateway

    @override
    async def run_inference(
        self,
        flow_id: FlowId,
        request: InferenceRequest,
    ) -> InferenceResult:

        flow_plan = self._front_end_plan_reader.get_current_flow_plan(flow_id)

        flow_plan

        return await self._worker_ingress_gateway.run_model_invocation(
            flow_id=flow_id,
            bundle=bundle,
        )

    @override
    async def get_local_worker_identity(self) -> WorkerId:
        return await self._worker_identity_gateway.get_worker_identity()
