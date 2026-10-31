from typing import override

from front_end.application.ports.inbound.inference_runner import (
    InferenceCoordinator,
)
from front_end.application.ports.outbound.front_end_plan_reader import (
    FrontEndPlanReader,
)
from front_end.application.ports.outbound.ingress_gateway import (
    IngressGateway,
)
from front_end.application.ports.outbound.sink_gateway import SinkGateway
from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.context.model_invocation_context import ModelInvocationContext
from front_end.domain.context.model_pass_context import ModelPassContext, ModelPassType
from front_end.domain.inference_result import InferenceRequest, InferenceResult
from front_end.domain.model_contract import ModelContract
from shared.flow.flow import FlowId


class DefaultInferenceRunner(InferenceCoordinator):
    def __init__(
        self,
        front_end_plan_reader: FrontEndPlanReader,
        ingress_gateway: IngressGateway,
        sink_gateway: SinkGateway,
    ) -> None:
        self._front_end_plan_reader = front_end_plan_reader
        self._ingress_gateway = ingress_gateway
        self._sink_gateway = sink_gateway

    @override
    async def get_current_contract_for_flow(
        self,
        flow_id: FlowId,
    ) -> ModelContract:
        plan_version = await self._front_end_plan_reader.get_current_plan_version()
        flow_plan = await self._front_end_plan_reader.get_flow_plan_by_version(
            flow_id, plan_version
        )

        return ModelContract(
            model_specs=flow_plan.model_specs, plan_version=plan_version
        )

    @override
    async def run_inference_for_request(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:

        curr_plan_version = await self._front_end_plan_reader.get_current_plan_version()
        if request.plan_version != curr_plan_version:
            raise ValueError("Plan has been updated, use the new one")
        flow_plan = await self._front_end_plan_reader.get_flow_plan_by_version(
            request.flow_id, curr_plan_version
        )

        flow_plan.validate_tensor_map(request.bundle)

        model_invocation_context = ModelInvocationContext(
            flow_id=request.flow_id,
        )

        ## NOTE: Here we should create a while loop for autoregression and use result yielding
        model_pass_context = ModelPassContext(
            model_invocation_context=model_invocation_context,
            plan_version=curr_plan_version,
            model_pass_type=ModelPassType.FORWARD,
        )

        external_source_invocation_context = ExternalSourceInvocationContext(
            model_pass_context,
        )

        await self._ingress_gateway.invoke_model(
            external_source_invocation_context,
            bundle=request.bundle,
        )

        result_map = await self._sink_gateway.retrieve_result(model_pass_context)

        return InferenceResult(result_map, request.request_id)
