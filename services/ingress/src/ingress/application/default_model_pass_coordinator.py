from typing import override

from ingress.application.abc.model_pass_coordinator import (
    ModelPassCoordinator,
    ModelPassRequestInfo,
)
from ingress.application.abc.result_collector import ResultCollector
from ingress.application.ports.outbound.model_pass_invoker import ModelPassInvoker
from ingress.application.ports.outbound.plan_store import PlanStore
from ingress.domain.model_pass_request import ModelPassRequest
from ingress.domain.model_pass_result import ModelPassResult
from shared.context.model_pass_context import ModelPassContext
from shared.tensor.tensor_bundle import TensorBundle


class DefaultModelPassCoordinator(ModelPassCoordinator):
    def __init__(
        self,
        invoker: ModelPassInvoker,
        plan_store: PlanStore,
        result_collector: ResultCollector,
    ) -> None:
        self._invoker = invoker
        self._result_collector = result_collector
        self._plan_store = plan_store

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

        plan = await self._plan_store.get_plan(model_pass_request_info.plan_version)

        await self._invoker.invoke(request=model_pass_request)

        result_bundle = await self._result_collector.collect_result(
            request_id=model_pass_context.model_pass_id,
            to_wait_list=plan.output_bindings,
        )

        return ModelPassResult(
            context=model_pass_context,
            bundle=result_bundle,
        )
