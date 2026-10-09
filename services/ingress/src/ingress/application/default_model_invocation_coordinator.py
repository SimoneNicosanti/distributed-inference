from typing import override

from ingress.application.abc.model_invocation_coordinator import (
    ModelInvocationCoordinator,
)
from ingress.application.abc.model_pass_coordinator import (
    ModelPassCoordinator,
    ModelPassRequestInfo,
)
from ingress.application.ports.outbound.plan_store import PlanStore
from ingress.domain.model_invocation_result import ModelInvocationResult
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassType
from shared.flow.flow import FlowId
from shared.service.service import ResultSinkId
from shared.tensor.tensor_bundle import TensorBundle


class DefaultModelInvocationCoordinator(ModelInvocationCoordinator):
    def __init__(
        self,
        result_sink_id: ResultSinkId,
        plan_store: PlanStore,
        model_pass_coordinator: ModelPassCoordinator,
    ) -> None:

        super().__init__()
        self._result_sink_id = result_sink_id
        self._plan_store = plan_store
        self._model_pass_coordinator = model_pass_coordinator

    @override
    async def run_model_invocation(
        self, flow_id: FlowId, bundle: TensorBundle
    ) -> ModelInvocationResult:

        ## Build request from request data
        model_invocation_context = ModelInvocationContext(
            flow_id=flow_id,
            result_sink=self._result_sink_id,
        )

        plan_version = await self._plan_store.get_plan_version_by_flow_id(flow_id)

        model_pass_request_info = ModelPassRequestInfo(
            model_invocation_context=model_invocation_context,
            plan_version=plan_version,
            model_pass_type=ModelPassType.FORWARD,
        )

        ## TODO: With multiple passes, we should add a while cycle until task compeletion
        model_pass_result = await self._model_pass_coordinator.run_model_pass(
            model_pass_request_info=model_pass_request_info,
            bundle=bundle,
        )

        return ModelInvocationResult(
            context=model_invocation_context,
            bundle=model_pass_result.bundle,
        )
