from typing import override

from worker.application.hop_contribution_snd.abc.hop_contribution_dispatcher import (
    HopContributionDispatcher,
)
from worker.application.hop_contribution_snd.abc.hop_contribution_propagator import (
    HopContributionPropagator,
)
from worker.application.hop_contribution_snd.abc.hop_output_router import (
    HopOutputRouter,
)
from worker.domain.hop.hop import InvocableHop
from worker.domain.hop.hop_invocation import HopInvocationResult
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class DefaultHopContributionPropagator(HopContributionPropagator):
    def __init__(
        self, router: HopOutputRouter, dispatcher: HopContributionDispatcher
    ) -> None:
        super().__init__()
        self._router = router
        self._dispatcher = dispatcher

    @override
    async def propagate(
        self,
        incoming_contributions: list[HopInvocationContribution],
        hop_invocation_result: HopInvocationResult[InvocableHop],
    ) -> None:

        routed_contributions = await self._router.route(
            incoming_contributions, hop_invocation_result
        )

        await self._dispatcher.dispatch(routed_contributions)
