from typing import override

from shared.context.model_pass_context import ModelPassContext
from worker.application.ingress.abc.ingress_input_router import IngressInputRouter
from worker.application.ingress.abc.model_pass_coordinator import (
    ModelPassCoordinator,
    ModelPassRequestInfo,
)
from worker.application.partition_invocation.output.abc.contribution_send_coordinator import (
    ContributionSendCoordinator,
)
from worker.application.ports.outbound.plan_store.plan_store_reader import (
    PlanStoreReader,
)
from worker.application.ports.outbound.sink.model_pass_result_gateway import (
    ModelPassResultRetrieverGateway,
)
from worker.domain.model_pass.model_pass_request import ModelPassRequest
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.tensor.tensor_bundle import TensorBundle


class DefaultModelPassCoordinator(ModelPassCoordinator):
    def __init__(
        self,
        input_router: IngressInputRouter,
        ingress_plan_reader: PlanStoreReader[IngressPlan],
        contribution_sender_coordinator: ContributionSendCoordinator,
        result_retriever_gateway: ModelPassResultRetrieverGateway,
    ) -> None:
        self._input_router = input_router
        self._ingress_plan_reader = ingress_plan_reader
        self._contribution_sender_coordinator = contribution_sender_coordinator
        self._result_retriever_gateway = result_retriever_gateway

    @override
    async def run_model_pass(
        self,
        model_pass_request_info: ModelPassRequestInfo,
        bundle: TensorBundle,
    ) -> ModelPassResult:

        model_pass_context = ModelPassContext(
            model_invocation_context=model_pass_request_info.model_invocation_context,
            plan_version=model_pass_request_info.plan_version,
            model_pass_type=model_pass_request_info.model_pass_type,
        )

        model_pass_request = ModelPassRequest(
            context=model_pass_context,
            bundle=bundle,
        )

        plan = await self._ingress_plan_reader.get_plan_by_version(
            model_pass_request_info.plan_version
        )
        flow_plan = plan.get_flow_plan_by_flow_id(
            flow_id=model_pass_request_info.flow_id
        )

        routed_contributions = await self._input_router.route(
            model_pass_request=model_pass_request,
            flow_plan=flow_plan,
        )
        await self._contribution_sender_coordinator.send_all_contributions(
            routed_contributions
        )

        return await self._result_retriever_gateway.retrieve_result(
            request_id=model_pass_context.model_pass_id,
            result_sink_id=flow_plan.sink_id,
        )
