import asyncio
from asyncio import Future
from typing import override

from shared.model.keys import TensorKey
from shared.plan.partition_replica_id import PartitionReplicaId
from worker.application.partition_invocation.input.abc.partition_invocation_contribution_inbox import (
    PartitionInvocationContributionInbox,
    ReadyPartitionInvocationRequest,
)
from worker.application.partition_invocation.input.abc.partition_invocation_request_assembler import (
    PartitionInvocationRequestAssembler,
)
from worker.application.partition_invocation.input.partition_invocation_collection_key import (
    PartitionInvocationCollectionKey,
)
from worker.application.ports.outbound.partition_invocation.store.partition_invocation_contribution_store import (
    PartitionInvocationContributionStore,
)
from worker.application.ports.outbound.plan_store.routing_plan_reader import (
    RoutingPlanReader,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)
from worker.domain.plan.routing_plan import RoutingPlan


class DefaultPartitionInvocationContributionInbox(PartitionInvocationContributionInbox):
    def __init__(
        self,
        routing_plan_reader: RoutingPlanReader,
        contribution_store: PartitionInvocationContributionStore,
        partition_invocation_request_assembler: PartitionInvocationRequestAssembler,
    ) -> None:
        super().__init__()
        self._routing_plan_reader = routing_plan_reader
        self._contribution_store = contribution_store
        self._collection_key_to_future: dict[
            PartitionInvocationCollectionKey, Future[None]
        ] = {}
        self._assembler = partition_invocation_request_assembler
        self._ready_queue: asyncio.Queue[PartitionInvocationCollectionKey] = (
            asyncio.Queue()
        )

    @override
    async def submit_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> Future[None]:
        plan_version = contribution.plan_version
        routing_plan = await self._routing_plan_reader.get_routing_plan_by_version(
            plan_version
        )
        if routing_plan is None:
            raise ValueError(f"Routing plan for version {plan_version} not found")

        collection_key = await self._contribution_store.put(contribution)
        if collection_key not in self._collection_key_to_future:
            self._collection_key_to_future[collection_key] = (
                asyncio.get_running_loop().create_future()
            )
        future = self._collection_key_to_future[collection_key]

        collected_contributions = await self._contribution_store.get(collection_key)

        if self._check_ready(collected_contributions, routing_plan):
            self._ready_queue.put_nowait(collection_key)

        return future

    @override
    async def next_ready(self) -> ReadyPartitionInvocationRequest:
        collection_key = await self._ready_queue.get()

        future = self._collection_key_to_future.pop(collection_key)
        contributions = await self._contribution_store.pop(collection_key)
        request = await self._assembler.assemble(contributions)

        return ReadyPartitionInvocationRequest(
            request=request,
            future=future,
            contributions=contributions,
        )

    @staticmethod
    def _check_ready(
        contributions: list[PartitionInvocationContribution],
        routing_plan: RoutingPlan,
    ) -> bool:
        replica_id = contributions[0].target_replica_id
        routing = routing_plan.get_routing_by_replica_id(replica_id)

        arrived_tensors_by_source: dict[PartitionReplicaId | None, set[TensorKey]] = {}

        for contribution in contributions:
            arrived_tensors_by_source.setdefault(
                contribution.source_replica_id,
                set(),
            ).update(contribution.bundle.get_tensor_names())

        return routing.check_inputs_ready(arrived_tensors_by_source)
