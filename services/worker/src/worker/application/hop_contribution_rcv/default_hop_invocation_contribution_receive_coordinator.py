from typing import override

from worker.application.hop_contribution_rcv.abc.contribution_pipeline import (
    ContributionPipeline,
)
from worker.application.ports.inbound.hop_contribution_rcv.hop_invocation_contribution_receive_coordinator import (
    HopInvocationContributionReceiveCoordinator,
)
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
    HopInvocationContributionAck,
)


class DefaultHopInvocationContributionReceiveCoordinator(
    HopInvocationContributionReceiveCoordinator
):
    def __init__(self, contribution_pipeline: ContributionPipeline) -> None:
        super().__init__()
        self._contribution_pipeline: ContributionPipeline = contribution_pipeline

    @override
    async def pass_hop_invocation_contribution(
        self, contribution: HopInvocationContribution
    ) -> HopInvocationContributionAck:
        handle = await self._contribution_pipeline.submit(contribution)

        await handle.future

        return HopInvocationContributionAck(
            source_context=contribution.source_context,
            destination_hop=contribution.destination_hop,
        )
