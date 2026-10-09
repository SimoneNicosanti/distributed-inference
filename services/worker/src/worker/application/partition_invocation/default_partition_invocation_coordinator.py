import asyncio
from contextlib import suppress
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from worker.application.partition_execution.abc.partition_execution_coordinator import (
    PartitionExecutionCoordinator,
)
from worker.application.partition_invocation.input.abc.partition_invocation_contribution_inbox import (
    PartitionInvocationContributionInbox,
)
from worker.application.partition_invocation.output.abc.contribution_send_coordinator import (
    ContributionSendCoordinator,
)
from worker.application.partition_invocation.output.abc.partition_output_router import (
    PartitionOutputRouter,
)
from worker.application.ports.inbound.partition_invocation.partition_invocation_coordinator import (
    PartitionInvocationCoordinator,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
    PartitionInvocationContributionAck,
)


class DefaultPartitionInvocationCoordinator(
    PartitionInvocationCoordinator, AsyncLifecycle
):
    def __init__(
        self,
        max_concurrent_invocations: int,
        contribution_inbox: PartitionInvocationContributionInbox,
        partition_execution_coordinator: PartitionExecutionCoordinator,
        output_router: PartitionOutputRouter,
        contribution_sender_coordinator: ContributionSendCoordinator,
    ):

        if max_concurrent_invocations <= 0:
            raise ValueError(
                f"max_concurrent_invocations must be a positive integer, got {max_concurrent_invocations}"
            )

        self._max_concurrent_invocations = max_concurrent_invocations
        self._contribution_inbox = contribution_inbox
        self._partition_execution_coordinator = partition_execution_coordinator
        self._output_router = output_router
        self._contribution_sender_coordinator = contribution_sender_coordinator

        self._supervisor_task: asyncio.Task[None] | None = None

    @override
    async def process_partition_invocation_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationContributionAck:

        completion_future = await self._contribution_inbox.submit_contribution(
            contribution
        )
        await asyncio.shield(completion_future)

        return PartitionInvocationContributionAck(
            context=contribution.source_invocation_context
        )

    async def _worker_loop(self) -> None:
        while True:
            ready_request = await self._contribution_inbox.next_ready()
            (
                request,
                future,
                contributions,
            ) = (
                ready_request.request,
                ready_request.future,
                ready_request.contributions,
            )

            partition_invocation_result = await self._partition_execution_coordinator.process_partition_invocation_request(
                request
            )

            routed_contributions = await self._output_router.route(
                contributions, partition_invocation_result
            )

            await self._contribution_sender_coordinator.send_all_contributions(
                routed_contributions
            )

            ## NOTE: We can move the set_result depending on when we want to notify the completion of the invocation
            future.set_result(None)

    async def _run_workers(self) -> None:
        async with asyncio.TaskGroup() as task_group:
            for index in range(self._max_concurrent_invocations):
                task_group.create_task(
                    self._worker_loop(),
                    name=f"partition-invocation-worker-{index}",
                )

    @override
    async def start(self) -> None:
        if self._supervisor_task is not None:
            return

        self._supervisor_task = asyncio.create_task(
            self._run_workers(),
            name="partition-invocation-supervisor",
        )

    @override
    async def stop(self) -> None:
        task = self._supervisor_task
        self._supervisor_task = None

        if task is None:
            return

        task.cancel()

        with suppress(asyncio.CancelledError):
            await task
