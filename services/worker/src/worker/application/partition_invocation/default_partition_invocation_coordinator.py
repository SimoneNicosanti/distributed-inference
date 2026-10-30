import asyncio
from contextlib import suppress
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from worker.application.contribution.abc.contribution_assembler import (
    ContributionAssembler,
)
from worker.application.contribution.abc.contribution_pipeline import (
    ContributionPipeline,
)
from worker.application.contribution.abc.inbox import Inbox
from worker.application.partition_execution.abc.partition_execution_coordinator import (
    PartitionExecutionCoordinator,
)
from worker.application.partition_invocation.input.partition_invocation_collection import (
    PartitionInvocationCollection,
    PartitionInvocationCollectionKey,
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
from worker.domain.partition.partition_invocation import PartitionInvocationRequest
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
    PartitionInvocationContributionAck,
)


class DefaultPartitionInvocationCoordinator(
    PartitionInvocationCoordinator, AsyncLifecycle
):
    def __init__(
        self,
        contribution_pipeline: ContributionPipeline[PartitionInvocationContribution],
        max_concurrent_invocations: int,
        inbox: Inbox[PartitionInvocationCollectionKey, PartitionInvocationCollection],
        assembler: ContributionAssembler[
            PartitionInvocationContribution, PartitionInvocationRequest
        ],
        partition_execution_coordinator: PartitionExecutionCoordinator,
        output_router: PartitionOutputRouter,
        contribution_sender_coordinator: ContributionSendCoordinator,
    ):
        self._contribution_pipeline = contribution_pipeline

        if max_concurrent_invocations <= 0:
            raise ValueError(
                f"max_concurrent_invocations must be a positive integer, got {max_concurrent_invocations}"
            )

        self._max_concurrent_invocations = max_concurrent_invocations

        self._inbox = inbox
        self._assembler = assembler
        self._partition_execution_coordinator = partition_execution_coordinator
        self._output_router = output_router
        self._contribution_sender_coordinator = contribution_sender_coordinator

        self._supervisor_task: asyncio.Task[None] | None = None

    @override
    async def process_partition_invocation_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationContributionAck:

        handle = await self._contribution_pipeline.submit(contribution)

        await handle.future

        return PartitionInvocationContributionAck(
            context=contribution.source_hop_invocation_context
        )

    async def _worker_loop(self) -> None:
        while True:
            partition_invocation_collection = await self._inbox.extract_next()
            contributions = list(partition_invocation_collection.contributions)

            partition_invocation_request = await self._assembler.assemble(contributions)

            partition_invocation_result = await self._partition_execution_coordinator.process_partition_invocation_request(
                partition_invocation_request
            )

            routed_contributions = await self._output_router.route(
                contributions, partition_invocation_result
            )

            await self._contribution_sender_coordinator.send_all_contributions(
                routed_contributions
            )

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
