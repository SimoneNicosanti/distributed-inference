import asyncio
from contextlib import suppress
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.application.hop_handler.abc.hop_handler import (
    HopHandler,
)
from worker.application.hop_handler.abc.partition_executor_registry import (
    PartitionExecutorRegistry,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.scheduling.abc.partition_invocation_request_scheduler import (
    HopInvocationRequestScheduler,
)
from worker.domain.activity.activity_request import ActivityType
from worker.domain.context.hop_execution_context import (
    HopExecutionContext,
)
from worker.domain.hop.hop import PartitionReplicaHop
from worker.domain.hop.hop_execution import (
    HopExecutionInput,
    HopExecutionOutput,
)
from worker.domain.hop.hop_invocation import (
    HopInvocationRequest,
    HopInvocationResult,
)


class DefaultPartitionReplicaHopHandler(
    HopHandler[PartitionReplicaHop], AsyncLifecycle
):
    def __init__(
        self,
        activity_request_factory: ActivityRequestFactory,
        activity_manager: ActivityManager,
        hop_invocation_request_scheduler: HopInvocationRequestScheduler,
        partition_executor_registry: PartitionExecutorRegistry,
    ) -> None:
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager
        self._hop_invocation_request_scheduler = hop_invocation_request_scheduler
        self._partition_executor_registry = partition_executor_registry

        self._controller_task: asyncio.Task[None] | None = None

    @override
    async def handle(
        self, hop_invocation_request: HopInvocationRequest[PartitionReplicaHop]
    ) -> HopInvocationResult[PartitionReplicaHop]:

        ## Here we can only enqueue the request
        ## The request will then be extracted in the loop of the coordinator
        ## and sent to the worker to be processed
        hop_invocation_result_future: asyncio.Future[
            HopInvocationResult[PartitionReplicaHop]
        ] = asyncio.get_running_loop().create_future()

        await self._hop_invocation_request_scheduler.enqueue(
            hop_invocation_request, hop_invocation_result_future
        )

        partition_invocation_result = await hop_invocation_result_future

        return partition_invocation_result

    async def _controller_loop(self) -> None:
        while True:
            (
                partition_invocation_request,
                hop_invocation_result_future,
            ) = await self._hop_invocation_request_scheduler.dequeue()

            activity_request = (
                self._activity_request_factory.create_request_for_activity_type(
                    ActivityType.EXECUTION_INFERENCE
                )
            )
            activity_grant = await self._activity_manager.acquire_activity_grant(
                activity_request
            )

            async with activity_grant:
                partition_invocation_result = await self._execute_partition_invocation(
                    partition_invocation_request
                )

            hop_invocation_result_future.set_result(partition_invocation_result)

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

        self._controller_task = None

    async def _execute_partition_invocation(
        self, hop_invocation_request: HopInvocationRequest[PartitionReplicaHop]
    ) -> HopInvocationResult[PartitionReplicaHop]:

        replica_id = hop_invocation_request.hop.partition_replica_id
        partition_execution_input = await self._build_partition_execution_input(
            hop_invocation_request
        )

        async with self._partition_executor_registry.acquire_partition_executor(
            replica_id
        ) as partition_executor:
            partition_execution_output = await partition_executor.execute(
                partition_execution_input
            )

        partition_invocation_result = await self._build_partition_invocation_result(
            hop_invocation_request, partition_execution_output
        )

        return partition_invocation_result

    ## Right now we are considering only stateless models: this means that the payload of the input is the same as the invocation
    ## In case of stateful models, we need to extract the output from the combined state + output
    async def _build_partition_invocation_result(
        self,
        partition_invocation_request: HopInvocationRequest[PartitionReplicaHop],
        partition_execution_output: HopExecutionOutput[PartitionReplicaHop],
    ) -> HopInvocationResult[PartitionReplicaHop]:

        partition_invocation_result = HopInvocationResult(
            context=partition_invocation_request.context,
            payload=partition_execution_output.payload,
        )

        return partition_invocation_result

    ## Right now we are considering only stateless models: this means that the payload of the input is the same as the invocation
    ## In case of stateful models, we need to add the state to the input as well
    async def _build_partition_execution_input(
        self, partition_invocation_request: HopInvocationRequest[PartitionReplicaHop]
    ) -> HopExecutionInput[PartitionReplicaHop]:
        partition_execution_context = HopExecutionContext(
            hop_invocation_context=partition_invocation_request.context
        )

        payload = partition_invocation_request.payload

        partition_execution_input = HopExecutionInput(
            context=partition_execution_context,
            payload=payload,
        )
        return partition_execution_input
