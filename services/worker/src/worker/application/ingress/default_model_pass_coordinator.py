from typing import override

from shared.context.model_pass_context import ModelPassContext
from shared.tensor.tensor_bundle import TensorBundle
from worker.application.ingress.abc.ingress_input_router import IngressInputRouter
from worker.application.ingress.abc.model_pass_coordinator import (
    ModelPassCoordinator,
    ModelPassRequestInfo,
)
from worker.application.ingress.abc.result_waiter import ResultWaiter
from worker.application.partition_invocation.output.abc.contribution_send_coordinator import (
    ContributionSendCoordinator,
)
from worker.application.ports.outbound.plan_store.ingress_plan_reader import (
    IngressPlanReader,
)
from worker.domain.model_pass.model_pass_request import ModelPassRequest
from worker.domain.model_pass.model_pass_result import ModelPassResult


class DefaultModelPassCoordinator(ModelPassCoordinator):
    def __init__(
        self,
        input_router: IngressInputRouter,
        ingress_plan_reader: IngressPlanReader,
        contribution_sender_coordinator: ContributionSendCoordinator,
        result_waiter: ResultWaiter,
    ) -> None:
        self._input_router = input_router
        self._ingress_plan_reader = ingress_plan_reader
        self._contribution_sender_coordinator = contribution_sender_coordinator
        self._result_waiter = result_waiter

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

        plan = await self._ingress_plan_reader.get_ingress_plan_by_version(
            model_pass_request_info.plan_version
        )
        if plan is None:
            raise ValueError(
                f"Ingress plan for version {model_pass_request_info.plan_version} not found"
            )

        flow_plan = plan.get_flow_plan_by_flow_id(
            flow_id=model_pass_request_info.flow_id
        )

        result_future = await self._result_waiter.register_result_wait(
            request_id=model_pass_context.model_pass_id,
            to_wait_list=list(flow_plan.result_tensors),
        )

        try:
            routed_contributions = await self._input_router.route(
                model_pass_request=model_pass_request,
                flow_plan=flow_plan,
            )

            await self._contribution_sender_coordinator.send_all_contributions(
                routed_contributions
            )
            result_bundle = await result_future
        finally:
            if not result_future.done():
                result_future.cancel()

        return ModelPassResult(
            context=model_pass_context,
            bundle=result_bundle,
        )
