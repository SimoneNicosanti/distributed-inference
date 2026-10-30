import asyncio
from typing import override

from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.application.hop_contribution_snd.abc.hop_contribution_dispatcher import (
    HopContributionDispatcher,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.hop_contribution_snd.contribution_sender import (
    ContributionSender,
)
from worker.domain.activity.activity_request import ActivityType
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)


class DefaultHopContributionDispatcher(HopContributionDispatcher):
    def __init__(
        self,
        activity_request_factory: ActivityRequestFactory,
        activity_manager: ActivityManager,
        contribution_sender: ContributionSender,
    ) -> None:
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager
        self._contribution_sender = contribution_sender

    @override
    async def dispatch(
        self,
        contributions: list[HopInvocationContribution],
    ) -> None:
        if not contributions:
            return

        activity_request = (
            self._activity_request_factory.create_request_for_activity_type(
                ActivityType.NETWORK_TRANSMISSION
            )
        )
        activity_grant = await self._activity_manager.acquire_activity_grant(
            activity_request
        )
        async with activity_grant, asyncio.TaskGroup() as task_group:
            for contribution in contributions:
                task_group.create_task(self._send_contribution(contribution))

    async def _send_contribution(
        self,
        contribution: HopInvocationContribution,
    ) -> None:
        await self._contribution_sender.send(contribution)
