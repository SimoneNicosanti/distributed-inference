from typing import override

from shared.model.keys import TensorKey
from shared.plan.partition_replica_id import PartitionReplicaId
from worker.application.contribution.abc.contribution_gathering_policy import (
    ContributionGatheringPolicy,
)
from worker.application.partition_invocation.input.partition_invocation_collection import (
    PartitionInvocationCollection,
    PartitionInvocationCollectionKey,
)
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    RoutingPlanReader,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionInvocationContributionGatheringPolicy(
    ContributionGatheringPolicy[
        PartitionInvocationCollectionKey,
        PartitionInvocationContribution,
        PartitionInvocationCollection,
    ]
):
    def __init__(
        self,
        routing_plan_reader: RoutingPlanReader,
    ) -> None:
        self._routing_plan_reader = routing_plan_reader

    @override
    async def is_ready(
        self,
        contributions: list[PartitionInvocationContribution],
    ) -> bool:
        if len(contributions) == 0:
            return False
        replica_id = contributions[0].target_replica_id
        plan_version = contributions[0].plan_version

        routing_plan = await self._routing_plan_reader.get_routing_plan_by_version(
            plan_version
        )
        routing = routing_plan.get_routing_by_replica_id(replica_id)
        arrived_tensors_by_source: dict[PartitionReplicaId | None, set[TensorKey]] = {}

        for contribution in contributions:
            arrived_tensors_by_source.setdefault(
                contribution.source_replica_id,
                set(),
            ).update(contribution.bundle.get_tensor_names())

        return routing.check_inputs_ready(arrived_tensors_by_source)

    @override
    async def gather(
        self,
        contributions: list[PartitionInvocationContribution],
    ) -> PartitionInvocationCollection:
        key = await self.key_of(contributions[0])
        collection = PartitionInvocationCollection(
            key=key, contributions=tuple(contributions)
        )
        return collection

    @override
    async def key_of(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationCollectionKey:
        return PartitionInvocationCollectionKey(
            model_pass_context=contribution.model_pass_context,
            partition_replica_id=contribution.target_replica_id,
        )

    @override
    async def validate(
        self,
        other_contributions: list[PartitionInvocationContribution],
        new_contribution: PartitionInvocationContribution,
    ) -> None:
        ## TODO: We should check that the input is coming from the right source
        ## If not, we should raise an error
        ## TODO: We should alse check that no duplicate is present

        ## TODO: Maybe something can be abstracted from is_ready too
        return
