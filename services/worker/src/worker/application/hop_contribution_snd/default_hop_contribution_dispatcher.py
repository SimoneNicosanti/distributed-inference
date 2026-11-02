import asyncio
from typing import override

from catalog.api.hop.hop_deployment_directory import (
    HopDeploymentDirectory,
)
from shared.service.service import WorkerId
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
        hop_deployment_directory: HopDeploymentDirectory,
        contribution_sender: ContributionSender,
    ) -> None:
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager
        self._hop_deployment_directory = hop_deployment_directory
        self._contribution_sender = contribution_sender

    @override
    async def dispatch(
        self,
        contributions: list[HopInvocationContribution],
    ) -> None:
        if not contributions:
            return

        worker_ids = await asyncio.gather(
            *[
                self._resolve_hop_replica_id(contribution)
                for contribution in contributions
            ]
        )

        activity_request = (
            self._activity_request_factory.create_request_for_activity_type(
                ActivityType.NETWORK_TRANSMISSION
            )
        )
        activity_grant = await self._activity_manager.acquire_activity_grant(
            activity_request
        )
        async with activity_grant, asyncio.TaskGroup() as task_group:
            for contribution, worker_id in zip(contributions, worker_ids, strict=True):
                task_group.create_task(
                    self._contribution_sender.send(contribution, worker_id)
                )

    async def _resolve_hop_replica_id(
        self, contribution: HopInvocationContribution
    ) -> WorkerId:
        worker_id = (
            await self._hop_deployment_directory.get_worker_for_planned_invocable_hop(
                plan_version=contribution.plan_version,
                planned_hop=contribution.destination_hop,
            )
        )
        return worker_id
