from typing import override
from uuid import uuid4

from shared.plan.plan import ServiceInferencePlan
from worker.application.deployment.abc.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.partition_invocation.input.abc.partition_invocation_contribution_collector import (
    PartitionInvocationContributionCollector,
)
from worker.application.partition_invocation.input.partition_invocation_collection_key import (
    PartitionInvocationCollectionKey,
)
from worker.application.ports.outbound.partition_invocation_contribution_store import (
    PartitionInvocationContributionStore,
)
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.domain.context.partition_invocation_context import (
    PartitionInvocationId,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class DefaultPartitionInvocationContributionCollector(
    PartitionInvocationContributionCollector, ServiceInferencePlanPreparer
):
    def __init__(
        self,
        service_inference_plan_store: ServiceInferencePlanStore,
        contribution_store: PartitionInvocationContributionStore,
    ):
        super().__init__()
        self._plan_store: ServiceInferencePlanStore = service_inference_plan_store
        self._contribution_store = contribution_store
        self._collection_key_to_invocation_ids: dict[
            PartitionInvocationCollectionKey, PartitionInvocationId
        ] = {}

    @override
    async def collect(
        self, contribution: PartitionInvocationContribution
    ) -> tuple[list[PartitionInvocationContribution] | None, PartitionInvocationId]:

        plan_version = contribution.plan_version
        plan = await self._plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(
                f"Service inference plan for version {plan_version} not found"
            )

        collection_key = await self._contribution_store.put(contribution)
        if collection_key not in self._collection_key_to_invocation_ids:
            partition_invocation_id = uuid4()
            self._collection_key_to_invocation_ids[collection_key] = (
                partition_invocation_id
            )
        partition_invocation_id = self._collection_key_to_invocation_ids[collection_key]

        collected_contributions = await self._contribution_store.get(collection_key)

        ready = self._check_ready(collected_contributions, plan)

        if ready:
            await self._clear_collection(collection_key)
            return collected_contributions, partition_invocation_id
        else:
            return None, partition_invocation_id

    async def _clear_collection(
        self, collection_key: PartitionInvocationCollectionKey
    ) -> None:
        self._collection_key_to_invocation_ids.pop(collection_key)
        await self._contribution_store.delete(collection_key)

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
