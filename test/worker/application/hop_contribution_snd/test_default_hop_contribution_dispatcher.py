from typing import override

import pytest

from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop import ExternalHopId
from shared.plan.hop_logical_id import PlannedIngressReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId, WorkerId
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
from worker.application.ports.outbound.plan_reader.worker_plan_view_readers import (
    HopPlacementPlanReader,
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
from worker.domain.plan.hop_placement_plan import HopPlacementPlan
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


class _HopPlacementPlanReader(HopPlacementPlanReader):
    def __init__(self, plan: HopPlacementPlan) -> None:
        self._plan = plan
        self.lookup: PlanVersion | None = None

    @override
    async def get_hop_placement_plan(self, version: PlanVersion) -> HopPlacementPlan:
        self.lookup = version
        return self._plan


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
                model_invocation_context=ModelInvocationContext(flow_id=FlowId()),
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
    placement_reader = _HopPlacementPlanReader(
        HopPlacementPlan(mapping={destination_hop: worker_id})
    )
    sender = _ContributionSender()
    dispatcher = DefaultHopContributionDispatcher(
        activity_request_factory=DefaultActivityRequestFactory(),
        activity_manager=activity_manager,
        hop_placement_plan_reader=placement_reader,
        contribution_sender=sender,
    )

    await dispatcher.dispatch([contribution])

    assert placement_reader.lookup == plan_version
    assert sender.sent == [(contribution, worker_id)]
    assert activity_manager.acquired_request is not None
    assert (
        activity_manager.acquired_request.activity_type
        is ActivityType.NETWORK_TRANSMISSION
    )
    assert activity_manager.released_grant_id is not None
