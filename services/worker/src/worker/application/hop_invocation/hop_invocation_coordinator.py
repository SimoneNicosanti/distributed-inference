import asyncio
from contextlib import suppress
from typing import cast, override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.plan.hop_logical_id import (
    IngressReplicaId,
    PartitionReplicaId,
    SinkReplicaId,
)
from worker.application.hop_contribution_rcv.abc.contribution_gathering_policy import (
    HopContributionCollection,
    HopContributionCollectionKey,
)
from worker.application.hop_contribution_snd.abc.hop_contribution_propagator import (
    HopContributionPropagator,
)
from worker.application.hop_handler.abc.hop_handler import (
    HopHandler,
)
from worker.application.hop_invocation.abc.contribution_assembler import (
    ContributionAssembler,
)
from worker.application.postbox.abc.inbox import Inbox
from worker.domain.hop.hop import InvocableHop
from worker.domain.hop.hop_invocation import HopInvocationRequest, HopInvocationResult


class HopInvocationCoordinator(AsyncLifecycle):
    def __init__(
        self,
        max_concurrent_invocations: int,
        inbox: Inbox[HopContributionCollectionKey, HopContributionCollection],
        assembler: ContributionAssembler,
        ingress_hop_handler: HopHandler[IngressReplicaId],
        partition_replica_hop_handler: HopHandler[PartitionReplicaId],
        sink_hop_handler: HopHandler[SinkReplicaId],
        contribution_propagator: HopContributionPropagator,
    ) -> None:

        if max_concurrent_invocations <= 0:
            raise ValueError(
                f"max_concurrent_invocations must be a positive integer, got {max_concurrent_invocations}"
            )

        self._max_concurrent_invocations = max_concurrent_invocations

        self._inbox = inbox
        self._assembler = assembler
        self._ingress_hop_handler = ingress_hop_handler
        self._partition_replica_hop_handler = partition_replica_hop_handler
        self._sink_hop_handler = sink_hop_handler
        self._contribution_propagator = contribution_propagator

        self._supervisor_task: asyncio.Task[None] | None = None

    async def coordination_loop(self) -> None:
        while True:
            hop_contribution_collection = await self._inbox.extract_next()
            contributions = list(hop_contribution_collection.collection)

            hop_invocation_request = await self._assembler.assemble(contributions)

            hop_invocation_result: (
                HopInvocationResult[PartitionReplicaId]
                | HopInvocationResult[IngressReplicaId]
                | HopInvocationResult[SinkReplicaId]
            )
            match hop_invocation_request.hop:
                case PartitionReplicaId():
                    hop_invocation_result = (
                        await self._partition_replica_hop_handler.handle(
                            cast(
                                HopInvocationRequest[PartitionReplicaId],
                                hop_invocation_request,
                            )
                        )
                    )

                case IngressReplicaId():
                    hop_invocation_result = await self._ingress_hop_handler.handle(
                        cast(
                            HopInvocationRequest[IngressReplicaId],
                            hop_invocation_request,
                        )
                    )

                case SinkReplicaId():
                    hop_invocation_result = await self._sink_hop_handler.handle(
                        cast(HopInvocationRequest[SinkReplicaId], hop_invocation_request)
                    )

                case hop:
                    raise TypeError(f"Unsupported hop: {hop!r}")

            await self._contribution_propagator.propagate(
                contributions,
                cast(HopInvocationResult[InvocableHop], hop_invocation_result),
            )

    async def _run_workers(self) -> None:
        async with asyncio.TaskGroup() as task_group:
            for index in range(self._max_concurrent_invocations):
                task_group.create_task(
                    self.coordination_loop(),
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
