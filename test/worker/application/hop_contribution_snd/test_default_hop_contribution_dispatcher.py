from typing import override

import pytest
from front_end.domain.context.model_invocation_context import ModelInvocationContext
from shared.plan.hop_deployment_id import HopDeploymentId, HopReplicaId
from worker.domain.context.model_pass_context import ModelPassType

from catalog.api.hop.hop_deployment_directory import (
    HopDeploymentDirectory,
)
from shared.context.model_pass_context import ModelPassContext
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop import ExternalHopId
from shared.plan.hop_logical_id import PlannedIngressReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId
from worker.application.activity.default_activity_request_factory import (
    DefaultActivityRequestFactory,
)
from worker.application.hop_contribution_snd.default_hop_contribution_dispatcher import (
    DefaultHopContributionDispatcher,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.hop_contribution_snd.contribution_sender import (
    ContributionSender,
)
from worker.domain.activity.activity_request import (
    ActivityGrant,
    ActivityGrantId,
    ActivityGrantInfo,
    ActivityRequest,
    ActivityType,
)
from worker.domain.context.hop_invocation_context import HopInvocationContext
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution
from worker.domain.tensor.tensor_bundle import TensorBundle


class _ActivityManager(ActivityManager):
    def __init__(self) -> None:
        self.acquired_request: ActivityRequest | None = None
        self.released_grant_id: ActivityGrantId | None = None

    @override
    async def acquire_activity_grant(
        self,
        request: ActivityRequest,
    ) -> ActivityGrant:
        self.acquired_request = request
        grant_info = ActivityGrantInfo(activity_grant_id=ActivityGrantId())
        return ActivityGrant(grant_info, self.release_activity_grant)

    @override
    async def release_activity_grant(
        self,
        activity_grant_id: ActivityGrantId,
    ) -> None:
        self.released_grant_id = activity_grant_id


class _DeploymentDirectory(HopDeploymentDirectory):
    def __init__(self, worker_id: WorkerId) -> None:
        self._worker_id = worker_id
        self.lookup: tuple[PlanVersion, HopReplicaId] | None = None

    @override
    async def get_worker_for_planned_invocable_hop[ReplicaIdT: HopReplicaId](
        self,
        plan_version: PlanVersion,
        replica_id: ReplicaIdT,
    ) -> HopDeploymentId[ReplicaIdT]:
        self.lookup = (plan_version, replica_id)
        return HopDeploymentId(replica_id=replica_id, worker_id=self._worker_id)


class _ContributionSender(ContributionSender):
    def __init__(self) -> None:
        self.sent: list[tuple[HopInvocationContribution, WorkerId]] = []

    @override
    async def send(
        self,
        contribution: HopInvocationContribution,
        worker_id: WorkerId,
    ) -> None:
        self.sent.append((contribution, worker_id))


def _contribution(
    plan_version: PlanVersion,
    destination_hop: PlannedIngressReplicaId,
) -> HopInvocationContribution:
    return HopInvocationContribution(
        source_context=HopInvocationContext(
            model_pass_context=ModelPassContext(
                model_invocation_context=ModelInvocationContext(
                    flow_id=FlowId(user_id=UserId())
                ),
                plan_version=plan_version,
                model_pass_type=ModelPassType.FORWARD,
            ),
            hop=ExternalHopId(),
        ),
        destination_hop=destination_hop,
        bundle=TensorBundle({}),
    )


@pytest.mark.asyncio
async def test_dispatch_resolves_worker_before_sending() -> None:
    plan_version = PlanVersion(version_number=1)
    destination_hop = PlannedIngressReplicaId(
        model_variant_id=ModelVariantId(
            model_id=ModelId("test-model"),
            variant_tag="test-variant",
        ),
        replica_idx=0,
    )
    contribution = _contribution(plan_version, destination_hop)
    worker_id = ServiceId()
    activity_manager = _ActivityManager()
    deployment_directory = _DeploymentDirectory(worker_id)
    sender = _ContributionSender()
    dispatcher = DefaultHopContributionDispatcher(
        activity_request_factory=DefaultActivityRequestFactory(),
        activity_manager=activity_manager,
        hop_deployment_directory=deployment_directory,
        contribution_sender=sender,
    )

    await dispatcher.dispatch([contribution])

    assert deployment_directory.lookup == (plan_version, destination_hop)
    assert sender.sent == [(contribution, worker_id)]
    assert activity_manager.acquired_request is not None
    assert (
        activity_manager.acquired_request.activity_type
        is ActivityType.NETWORK_TRANSMISSION
    )
    assert activity_manager.released_grant_id is not None
