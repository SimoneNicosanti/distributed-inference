import asyncio
from typing import override

from distributed_inference.activity_manager.application.ports.inbound.activity_manager import (
    ActivityManager,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
    ActivityType,
)
from distributed_inference.activity_manager.domain.resource_type import (
    ResourceRequirement,
    ResourceType,
)
from distributed_inference.building_blocks.lifecycle.async_lifecycle import (
    AsyncLifecycle,
)
from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from distributed_inference.worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from distributed_inference.worker.application.scheduling.contracts.sub_model_invocation_request_scheduler import (
    SubModelInvocationRequestScheduler,
)
from distributed_inference.worker.application.sub_model_execution.contracts.sub_model_execution_coordinator import (
    SubModelExecutionCoordinator,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
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
    ) -> None:
        self._inference_plan_store = inference_plan_store
        self._activity_manager = activity_manager
        self._sub_model_inference_request_scheduler = (
            sub_model_inference_request_scheduler
        )

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
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
        future: asyncio.Future[SubModelInvocationResponse] = (
            asyncio.get_running_loop().create_future()
        )

        await self._sub_model_inference_request_scheduler.enqueue(
            sub_model_invocation_request, future
        )

        inference_response: SubModelInvocationResponse = await future

        return inference_response

    @override
    async def start(self) -> None:
        while True:
            (
                inference_request,
                inference_response_future,
            ) = await self._sub_model_inference_request_scheduler.dequeue()

            activity_grant = await self._activity_manager.request_activity_grant(
                self._build_activity_request()
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
            activity_resources={
                ResourceType.COMPUTE: ResourceRequirement(quantity=0, exclusive=True)
            },
        )
        return activity_request

    async def _actual_process_inference_request(
        self, inference_request: SubModelInvocationRequest
    ) -> SubModelInvocationResponse:
        raise NotImplementedError
