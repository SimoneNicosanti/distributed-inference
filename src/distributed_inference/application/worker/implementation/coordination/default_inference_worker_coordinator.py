import asyncio
from typing import override

from distributed_inference.application.lifecycle.contracts.async_lifecycle import (
    AsyncLifecycle,
)
from distributed_inference.application.worker.contracts.activity.activity_request import (
    ActivityRequest,
    ActivityType,
)
from distributed_inference.application.worker.contracts.activity.activity_request_scheduler import (
    ActivityRequestScheduler,
)
from distributed_inference.application.worker.contracts.deployment.inference_plan_preparer import (
    InferencePlanPreparer,
)
from distributed_inference.application.worker.contracts.deployment.inference_plan_store import (
    InferencePlanStore,
)
from distributed_inference.application.worker.contracts.execution.inference.inference_request_scheduler import (
    InferenceRequestScheduler,
)
from distributed_inference.application.worker.contracts.execution.inference.inference_worker_coordinator import (
    InferenceWorkerCoordinator,
)
from distributed_inference.application.worker.contracts.resource.resource_type import (
    ResourceType,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceRequest,
    InferenceResponse,
)
from distributed_inference.domain.plan import ServiceInferencePlan


class DefaultInferenceWorkerCoordinator(
    InferenceWorkerCoordinator, InferencePlanPreparer, AsyncLifecycle
):
    def __init__(
        self,
        inference_plan_store: InferencePlanStore,
        activity_scheduler: ActivityRequestScheduler,
        inference_request_scheduler: InferenceRequestScheduler,
    ) -> None:
        self._inference_plan_store = inference_plan_store
        self._node_activity_scheduler = activity_scheduler
        self._inference_request_scheduler = inference_request_scheduler

    @override
    async def prepare_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## 1. Create workers based on new plan
        ## 2. Do not delete existing workers, but keep them for possible late requests
        ## 3. Update active plan
        raise NotImplementedError

    @override
    async def process_inference_request(
        self, inference_request: InferenceRequest
    ) -> InferenceResponse:

        ## Here we can only enqueue the request
        ## The request will then be extracted in the loop of the coordinator
        ## and sent to the worker to be processed
        future: asyncio.Future[InferenceResponse] = (
            asyncio.get_running_loop().create_future()
        )

        await self._inference_request_scheduler.enqueue(inference_request, future)

        inference_response: InferenceResponse = await future

        return inference_response

    @override
    async def start(self) -> None:
        while True:
            (
                inference_request,
                inference_response_future,
            ) = await self._inference_request_scheduler.dequeue()

            await self._wait_for_run_permission()

            inference_response = await self._actual_process_inference_request(
                inference_request
            )

            inference_response_future.set_result(inference_response)

        raise NotImplementedError

    @override
    async def stop(self) -> None:
        raise NotImplementedError

    async def _wait_for_run_permission(self) -> None:
        can_run_future = asyncio.get_running_loop().create_future()

        activity_request = ActivityRequest(
            type=ActivityType.INFERENCE_EXECUTION,
            resource_type=ResourceType.COMPUTE,
        )
        await self._node_activity_scheduler.enqueue(activity_request, can_run_future)
        await can_run_future

    async def _actual_process_inference_request(
        self, inference_request: InferenceRequest
    ) -> InferenceResponse:
        raise NotImplementedError
