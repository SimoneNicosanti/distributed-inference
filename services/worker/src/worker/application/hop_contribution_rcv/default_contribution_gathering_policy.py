from typing import override

from shared.model.keys import TensorKey
from worker.application.hop_contribution_rcv.abc.contribution_gathering_policy import (
    ContributionGatheringPolicy,
    HopContributionCollection,
    HopContributionCollectionKey,
)
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    HopRoutingPlanReader,
)
from worker.domain.hop.hop import SourceHop
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)
from worker.domain.plan.hop_routing_plan import HopAddress


class DefaultContributionGatheringPolicy(ContributionGatheringPolicy):
    def __init__(
        self,
        routing_plan_reader: HopRoutingPlanReader,
    ) -> None:
        self._routing_plan_reader = routing_plan_reader

    @override
    async def key_of(
        self, contribution: HopInvocationContribution
    ) -> HopContributionCollectionKey:
        return HopContributionCollectionKey(
            model_pass_context=contribution.model_pass_context,
            hop=contribution.destination_hop,
        )

    @override
    async def gather(
        self,
        contributions: list[HopInvocationContribution],
    ) -> HopContributionCollection:
        key = await self.key_of(contributions[0])
        collection = HopContributionCollection(key=key, collection=tuple(contributions))
        return collection

    @override
    async def is_ready(
        self,
        contributions: list[HopInvocationContribution],
    ) -> bool:
        if not contributions:
            return False

        first = contributions[0]
        key = await self.key_of(first)
        if any(
            await self.key_of(contribution) != key
            for contribution in contributions[1:]  # type: ignore
        ):
            raise ValueError("Contributions belong to different collections")

        routing_plan = await self._routing_plan_reader.get_hop_routing_plan_by_version(
            first.plan_version
        )
        destination = HopAddress(
            flow_id=first.flow_id,
            hop=first.destination_hop,
        )
        routings = routing_plan.get_routings_for_hop_address(destination)

        arrived: dict[HopAddress[SourceHop], set[TensorKey]] = {}
        for contribution in contributions:
            source = HopAddress(
                flow_id=contribution.flow_id,
                hop=contribution.source_hop,
            )
            arrived.setdefault(source, set()).update(
                contribution.bundle.get_tensor_names()
            )

        return routings.check_inputs_ready(arrived)

    @override
    async def validate(
        self,
        other_contributions: list[HopInvocationContribution],
        new_contribution: HopInvocationContribution,
    ) -> None:
        key = await self.key_of(new_contribution)
        if any(
            await self.key_of(contribution) != key
            for contribution in other_contributions  # type: ignore
        ):
            raise ValueError("Contributions belong to different collections")

        routing_plan = await self._routing_plan_reader.get_hop_routing_plan_by_version(
            new_contribution.plan_version
        )
        flow_id = new_contribution.flow_id
        destination = HopAddress(
            flow_id=flow_id,
            hop=new_contribution.destination_hop,
        )
        routings = routing_plan.get_routings_for_hop_address(destination)

        source = HopAddress(
            flow_id=flow_id,
            hop=new_contribution.source_hop,
        )
        new_tensors = new_contribution.bundle.get_tensor_names()
        routings.check_inputs_ready({source: new_tensors})

        received_tensors = {
            tensor
            for contribution in other_contributions
            for tensor in contribution.bundle.get_tensor_names()
        }
        duplicates = new_tensors & received_tensors
        if duplicates:
            raise ValueError(f"Duplicate input tensors: {duplicates}")
