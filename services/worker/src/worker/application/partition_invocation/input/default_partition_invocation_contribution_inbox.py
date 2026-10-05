import asyncio
from asyncio import Future
from typing import override

from shared.plan.plan import ServiceInferencePlan
from worker.application.deployment.abc.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
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
from worker.application.ports.outbound.partition_invocation.partition_invocation_contribution_store import (
    PartitionInvocationContributionStore,
)
from worker.application.ports.outbound.plan_store.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class DefaultPartitionInvocationContributionInbox(
    PartitionInvocationContributionInbox, ServiceInferencePlanPreparer
):
    def __init__(
        self,
        service_inference_plan_store: ServiceInferencePlanStore,
        contribution_store: PartitionInvocationContributionStore,
        partition_invocation_request_assembler: PartitionInvocationRequestAssembler,
    ):
        super().__init__()
        self._plan_store: ServiceInferencePlanStore = service_inference_plan_store
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
        plan = await self._plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(
                f"Service inference plan for version {plan_version} not found"
            )

        collection_key = await self._contribution_store.put(contribution)
        if collection_key not in self._collection_key_to_future:
            self._collection_key_to_future[collection_key] = (
                asyncio.get_running_loop().create_future()
            )
        future = self._collection_key_to_future[collection_key]

        collected_contributions = await self._contribution_store.get(collection_key)

        ready = self._check_ready(collected_contributions, plan)

        if ready:
            self._ready_queue.put_nowait(collection_key)

        return future

    @override
    async def next_ready(
        self,
    ) -> ReadyPartitionInvocationRequest:
        collection_key = await self._ready_queue.get()

        future = self._collection_key_to_future.pop(collection_key)
        contributions = await self._contribution_store.pop(collection_key)
        request = await self._assembler.assemble(contributions)

        ready_request = ReadyPartitionInvocationRequest(
            request=request, future=future, contributions=contributions
        )

        return ready_request

    def _check_ready(
        self,
        contributions: list[PartitionInvocationContribution],
        service_inference_plan: ServiceInferencePlan,
    ) -> bool:

        partition_id = contributions[0].partition_id

        arrived_tensors: set[str] = set()
        for contribution in contributions:
            arrived_tensors.update(contribution.bundle.bundle.keys())

        execution_scheme = service_inference_plan.sub_model_execution_schemes[
            partition_id
        ]
        skip_scheme = service_inference_plan.sub_model_skip_schemes[partition_id]

        ## TODO: To do a better check, we should check that the tensors arrived from the expected sources according with the topology expressed in the plan
        ## TODO : check this in case of model split changes
        return set(arrived_tensors) == set(execution_scheme.inputs).union(
            skip_scheme.skip_tensors
        )

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## No need to do anything, we just check the plans in the store
        pass
