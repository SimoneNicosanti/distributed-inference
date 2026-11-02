from typing import override

from catalog.api.hop.hop_deployment_directory import HopDeploymentDirectory
from front_end.application.ports.inbound.inference_runner import (
    InferenceCoordinator,
)
from front_end.application.ports.outbound.inference.flow_plan_reader import (
    FlowPlanReader,
)
from front_end.application.ports.outbound.inference.ingress_gateway import (
    IngressGateway,
)
from front_end.application.ports.outbound.inference.sink_gateway import SinkGateway
from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.inference_result import InferenceRequest, InferenceResult
from front_end.domain.model_contract import ModelContract
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.flow.flow import FlowId


class DefaultInferenceRunner(InferenceCoordinator):
    def __init__(
        self,
        flow_plan_reader: FlowPlanReader,
        hop_deployment_directory: HopDeploymentDirectory,
        ingress_gateway: IngressGateway,
        sink_gateway: SinkGateway,
    ) -> None:
        self._flow_plan_reader = flow_plan_reader
        self._hop_deployment_directory = hop_deployment_directory
        self._ingress_gateway = ingress_gateway
        self._sink_gateway = sink_gateway

    @override
    async def get_current_contract_for_flow(
        self,
        flow_id: FlowId,
    ) -> ModelContract:
        plan_version = await self._flow_plan_reader.get_current_plan_version_for_flow(
            flow_id
        )
        flow_plan = await self._flow_plan_reader.get_flow_plan_by_plan_version(
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

        curr_plan_version = (
            await self._flow_plan_reader.get_current_plan_version_for_flow(
                request.flow_id
            )
        )
        if request.plan_version != curr_plan_version:
            raise ValueError("Plan has been updated, use the new one")
        flow_plan = await self._flow_plan_reader.get_flow_plan_by_plan_version(
            request.flow_id, curr_plan_version
        )

        flow_plan.validate_tensor_map(request.bundle)

        model_invocation_context = ModelInvocationContext(
            flow_id=request.flow_id, model_invocation_id=request.request_id
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

        ingress_worker_id = (
            await self._hop_deployment_directory.get_worker_for_planned_invocable_hop(
                curr_plan_version, flow_plan.ingress_replica_id
            )
        )

        sink_worker_id = (
            await self._hop_deployment_directory.get_worker_for_planned_invocable_hop(
                curr_plan_version, flow_plan.sink_replica_id
            )
        )

        await self._ingress_gateway.submit_model_pass(
            external_source_invocation_context,
            bundle=request.bundle,
            planned_ingress_replica_id=flow_plan.ingress_replica_id,
            worker_id=ingress_worker_id,
        )

        result_map = await self._sink_gateway.retrieve_result(
            model_pass_context, sink_worker_id
        )

        return InferenceResult(result_map, request.request_id)
