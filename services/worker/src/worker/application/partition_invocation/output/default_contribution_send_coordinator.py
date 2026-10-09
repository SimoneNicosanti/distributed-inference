import asyncio
from typing import override

from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.application.partition_invocation.output.abc.contribution_send_coordinator import (
    ContributionSendCoordinator,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.partition_invocation.sender.contribution_sender import (
    ContributionSender,
)
from worker.domain.activity.activity_request import ActivityType
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class DefaultContributionSendCoordinator(ContributionSendCoordinator):
    def __init__(
        self,
        activity_request_factory: ActivityRequestFactory,
        activity_manager: ActivityManager,
        partition_invocation_contribution_sender: ContributionSender[
            PartitionInvocationContribution
        ],
        model_pass_result_contribution_sender: ContributionSender[
            ModelPassResultContribution
        ],
    ) -> None:
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager
        self._part_contribution_sender = partition_invocation_contribution_sender
        self._model_pass_contribution_sender = model_pass_result_contribution_sender

    @override
    async def send_all_contributions(
        self,
        all_contributions: list[
            PartitionInvocationContribution | ModelPassResultContribution
        ],
    ) -> None:
        activity_request = (
            self._activity_request_factory.create_request_for_activity_type(
                ActivityType.NETWORK_TRANSMISSION
            )
        )
        activity_grant = await self._activity_manager.acquire_activity_grant(
            activity_request
        )
        async with activity_grant, asyncio.TaskGroup() as task_group:
            for contribution in all_contributions:
                task_group.create_task(self._send_contribution(contribution))

    async def _send_contribution(
        self,
        contribution: PartitionInvocationContribution | ModelPassResultContribution,
    ) -> None:
        if isinstance(contribution, PartitionInvocationContribution):
            await self._part_contribution_sender.send(contribution)
            return

        if isinstance(contribution, ModelPassResultContribution):
            await self._model_pass_contribution_sender.send(contribution)
            return

        raise TypeError(f"Invalid contribution type: {type(contribution)}")
