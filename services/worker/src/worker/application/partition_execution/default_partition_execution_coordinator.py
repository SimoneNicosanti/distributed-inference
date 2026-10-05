import asyncio
from contextlib import suppress
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.plan.plan import ServiceInferencePlan
from worker.application.deployment.abc.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.partition_execution.abc.partition_execution_coordinator import (
    PartitionExecutionCoordinator,
)
from worker.application.partition_execution.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.plan_store.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.application.scheduling.abc.partition_invocation_request_scheduler import (
    PartitionInvocationRequestScheduler,
)
from worker.domain.activity.activity_request import (
    ActivityRequest,
    ActivityType,
    ResourceRequirement,
    ResourceType,
)
from worker.domain.context.partition_execution_context import (
    PartitionExecutionContext,
)
from worker.domain.partition.partition_execution import (
    PartitionExecutionInput,
    PartitionExecutionOutput,
)
from worker.domain.partition.partition_invocation import (
    PartitionInvocationRequest,
    PartitionInvocationResult,
)


class DefaultPartitionExecutionCoordinator(
    PartitionExecutionCoordinator, ServiceInferencePlanPreparer, AsyncLifecycle
):
    def __init__(
        self,
        inference_plan_store: ServiceInferencePlanStore,
        activity_manager: ActivityManager,
        partition_invocation_request_scheduler: PartitionInvocationRequestScheduler,
        partition_executor_registry: PartitionExecutorRegistry,
    ) -> None:
        self._inference_plan_store = inference_plan_store
        self._activity_manager = activity_manager
        self._partition_invocation_request_scheduler = (
            partition_invocation_request_scheduler
        )
        self._partition_executor_registry = partition_executor_registry

        self._controller_task: asyncio.Task[None] | None = None

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## TODO: Check this flow; it is probably different
        ## 1. Create workers based on new plan
        ## 2. Do not delete existing workers, but keep them for possible late requests
        ## 3. Update active plan
        pass

    @override
    async def process_partition_invocation_request(
        self, partition_invocation_request: PartitionInvocationRequest
    ) -> PartitionInvocationResult:

        ## Here we can only enqueue the request
        ## The request will then be extracted in the loop of the coordinator
        ## and sent to the worker to be processed
        partition_invocation_result_future: asyncio.Future[
            PartitionInvocationResult
        ] = asyncio.get_running_loop().create_future()

        await self._partition_invocation_request_scheduler.enqueue(
            partition_invocation_request, partition_invocation_result_future
        )

        partition_invocation_result = await partition_invocation_result_future

        return partition_invocation_result

    async def _controller_loop(self) -> None:
        while True:
            (
                partition_invocation_request,
                partition_invocation_result_future,
            ) = await self._partition_invocation_request_scheduler.dequeue()

            activity_request = self._build_activity_request()
            activity_grant = await self._activity_manager.acquire_activity_grant(
                activity_request
            )

            async with activity_grant:
                partition_invocation_result = await self._execute_partition_invocation(
                    partition_invocation_request
                )

            partition_invocation_result_future.set_result(partition_invocation_result)

    @override
    async def start(self) -> None:
        if self._controller_task is not None:
            return

        self._controller_task = asyncio.create_task(
            self._controller_loop(),
            name="partition-invocation-coordinator",
        )

    @override
    async def stop(self) -> None:
        if self._controller_task is None:
            return

        self._controller_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._controller_task

    def _build_activity_request(self) -> ActivityRequest:
        activity_request = ActivityRequest(
            activity_type=ActivityType.INFERENCE_EXECUTION,
            resource_requirements={
                ResourceType.COMPUTE: ResourceRequirement(quantity=0, exclusive=True)
            },
        )
        return activity_request

    async def _execute_partition_invocation(
        self, partition_invocation_request: PartitionInvocationRequest
    ) -> PartitionInvocationResult:

        partition_deployment = partition_invocation_request.partition_deployment
        partition_execution_input = await self._build_partition_execution_input(
            partition_invocation_request
        )

        async with self._partition_executor_registry.acquire_partition_executor(
            partition_deployment
        ) as partition_executor:
            partition_execution_output = await partition_executor.execute(
                partition_execution_input
            )

        partition_invocation_result = await self._build_partition_invocation_result(
            partition_invocation_request, partition_execution_output
        )

        return partition_invocation_result

    ## Right now we are considering only stateless models: this means that the payload of the input is the same as the invocation
    ## In case of stateful models, we need to extract the output from the combined state + output
    async def _build_partition_invocation_result(
        self,
        partition_invocation_request: PartitionInvocationRequest,
        partition_execution_output: PartitionExecutionOutput,
    ) -> PartitionInvocationResult:

        partition_invocation_result = PartitionInvocationResult(
            context=partition_invocation_request.context,
            payload=partition_execution_output.payload,
        )

        return partition_invocation_result

    ## Right now we are considering only stateless models: this means that the payload of the input is the same as the invocation
    ## In case of stateful models, we need to add the state to the input as well
    async def _build_partition_execution_input(
        self, partition_invocation_request: PartitionInvocationRequest
    ) -> PartitionExecutionInput:
        partition_execution_context = PartitionExecutionContext(
            partition_invocation_context=partition_invocation_request.context
        )

        payload = partition_invocation_request.payload

        partition_execution_input = PartitionExecutionInput(
            partition_execution_context=partition_execution_context,
            payload=payload,
        )
        return partition_execution_input
