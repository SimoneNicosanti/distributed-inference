import asyncio
from typing import override

from integration.plan.plan import ServiceInferencePlan
from lifecycle.async_lifecycle import AsyncLifecycle
from worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.ports.outbound.activity_manager import ActivityManager
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.application.scheduling.contracts.sub_model_invocation_request_scheduler import (
    SubModelInvocationRequestScheduler,
)
from worker.application.sub_model_execution.contracts.sub_model_execution_coordinator import (
    SubModelExecutionCoordinator,
)
from worker.application.sub_model_execution.contracts.sub_model_executor_registry import (
    SubModelExecutorRegistry,
)
from worker.domain.activity.activity_request import (
    ActivityRequest,
    ActivityType,
    ResourceRequirement,
    ResourceType,
)
from worker.domain.sub_model.execution.sub_model_execution_context import (
    SubModelExecutionContext,
)
from worker.domain.sub_model.execution.sub_model_execution_input_output import (
    SubModelExecutionInput,
    SubModelExecutionOutput,
)
from worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
    SubModelInvocationResponse,
)


class DefaultSubModelExecutionCoordinator(
    SubModelExecutionCoordinator, ServiceInferencePlanPreparer, AsyncLifecycle
):
    def __init__(
        self,
        inference_plan_store: ServiceInferencePlanStore,
        activity_manager: ActivityManager,
        sub_model_inference_request_scheduler: SubModelInvocationRequestScheduler,
        sub_model_executor_registry: SubModelExecutorRegistry,
    ) -> None:
        self._inference_plan_store = inference_plan_store
        self._activity_manager = activity_manager
        self._sub_model_inference_request_scheduler = (
            sub_model_inference_request_scheduler
        )
        self._sub_model_executor_registry = sub_model_executor_registry

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## TODO: Check this flow; it is probably different
        ## 1. Create workers based on new plan
        ## 2. Do not delete existing workers, but keep them for possible late requests
        ## 3. Update active plan
        raise NotImplementedError

    @override
    async def process_sub_model_invocation_request(
        self, sub_model_invocation_request: SubModelInvocationRequest
    ) -> SubModelInvocationResponse:

        ## Here we can only enqueue the request
        ## The request will then be extracted in the loop of the coordinator
        ## and sent to the worker to be processed
        inference_response_future: asyncio.Future[SubModelInvocationResponse] = (
            asyncio.get_running_loop().create_future()
        )

        await self._sub_model_inference_request_scheduler.enqueue(
            sub_model_invocation_request, inference_response_future
        )

        inference_response: SubModelInvocationResponse = await inference_response_future

        return inference_response

    @override
    async def start(self) -> None:
        while True:
            (
                inference_request,
                inference_response_future,
            ) = await self._sub_model_inference_request_scheduler.dequeue()

            activity_request = self._build_activity_request()
            activity_grant = await self._activity_manager.acquire_activity_grant(
                activity_request
            )

            async with activity_grant:
                inference_response = await self._actual_process_inference_request(
                    inference_request
                )

            inference_response_future.set_result(inference_response)

    @override
    async def stop(self) -> None:
        raise NotImplementedError

    def _build_activity_request(self) -> ActivityRequest:
        activity_request = ActivityRequest(
            activity_type=ActivityType.INFERENCE_EXECUTION,
            required_locks={
                ResourceType.COMPUTE: ResourceRequirement(quantity=0, exclusive=True)
            },
        )
        return activity_request

    async def _actual_process_inference_request(
        self, invocation_request: SubModelInvocationRequest
    ) -> SubModelInvocationResponse:

        sub_model_deployment_id = invocation_request.context.sub_model_deployment_id
        sub_model_execution_input = await self._build_sub_model_execution_input(
            invocation_request
        )

        async with self._sub_model_executor_registry.acquire_sub_model_executor(
            sub_model_deployment_id
        ) as sub_model_executor:
            sub_model_execution_output = (
                await sub_model_executor.process_sub_model_inference_input(
                    sub_model_execution_input
                )
            )

        sub_model_invocation_response = await self._build_sub_model_execution_response(
            invocation_request, sub_model_execution_output
        )

        return sub_model_invocation_response

    async def _build_sub_model_execution_response(
        self,
        invocation_request: SubModelInvocationRequest,
        sub_model_execution_output: SubModelExecutionOutput,
    ) -> SubModelInvocationResponse:

        plan_version = invocation_request.plan_version
        plan = await self._inference_plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(f"Plan version {plan_version} does not exist")
        sub_model_id = invocation_request.sub_model_id
        sub_model_outputs = plan.sub_model_execution_schemes[sub_model_id].outputs

        filered_payload = invocation_request.payload.filter(sub_model_outputs).merge(
            sub_model_execution_output.payload.filter(sub_model_outputs)
        )
        if set(filered_payload.bundle.keys()) != set(sub_model_outputs):
            raise ValueError(
                f"Sub-model {sub_model_id} expects {len(sub_model_outputs)} outputs, but received {len(filered_payload.bundle)}"
            )

        sub_model_invocation_response = SubModelInvocationResponse(
            context=invocation_request.context,
            payload=filered_payload,
        )

        return sub_model_invocation_response

    async def _build_sub_model_execution_input(
        self, invocation_request: SubModelInvocationRequest
    ) -> SubModelExecutionInput:
        sub_model_execution_context = SubModelExecutionContext(
            sub_model_invocation_context=invocation_request.context
        )

        plan_version = invocation_request.plan_version
        plan = await self._inference_plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(f"Plan version {plan_version} does not exist")
        sub_model_id = invocation_request.sub_model_id
        sub_model_inputs = plan.sub_model_execution_schemes[sub_model_id].inputs

        filtered_payload = invocation_request.payload.filter(sub_model_inputs)

        if set(filtered_payload.bundle.keys()) != set(sub_model_inputs):
            raise ValueError(
                f"Sub-model {sub_model_id} expects {len(sub_model_inputs)} inputs, but received {len(filtered_payload.bundle)}"
            )

        sub_model_execution_input = SubModelExecutionInput(
            sub_model_execution_context=sub_model_execution_context,
            payload=filtered_payload,
        )
        return sub_model_execution_input
