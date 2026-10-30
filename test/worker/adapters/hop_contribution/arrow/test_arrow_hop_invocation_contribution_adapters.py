import asyncio
from threading import Thread
from typing import override

import numpy as np
import pytest

from integration.service_instance.service_instance import (
    ArrowUrl,
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.service_instance.worker_instance import WorkerInstance
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.inbound.hop_contribution.arrow.arrow_hop_invocation_contribution_server import (
    ArrowHopInvocationContributionServer,
)
from worker.adapters.outbound.hop_contribution.arrow.arrow_hop_invocation_contribution_sender import (
    ArrowHopInvocationContributionSender,
)
from worker.application.ports.inbound.hop_contribution_rcv.hop_invocation_contribution_receive_coordinator import (
    HopInvocationContributionReceiveCoordinator,
)
from worker.domain.context.hop_invocation_context import HopInvocationContext
from worker.domain.context.model_invocation_context import ModelInvocationContext
from worker.domain.context.model_pass_context import ModelPassContext, ModelPassType
from worker.domain.hop.hop import ExternalHop, IngressHop
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
    HopInvocationContributionAck,
)
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


def _contribution(destination_id: ServiceId) -> HopInvocationContribution:
    return HopInvocationContribution(
        source_context=HopInvocationContext(
            model_pass_context=ModelPassContext(
                model_invocation_context=ModelInvocationContext(
                    flow_id=FlowId(user_id=UserId())
                ),
                plan_version=PlanVersion(version_number=1),
                model_pass_type=ModelPassType.FORWARD,
            ),
            hop=ExternalHop(),
        ),
        destination_hop=IngressHop(ingress_id=destination_id),
        bundle=TensorBundle(
            {"input": Tensor.from_numpy(np.array([[1, 2]], dtype=np.float32))}
        ),
    )


def test_contribution_wire_round_trip() -> None:
    contribution = _contribution(ServiceId())

    batch = ArrowHopInvocationContributionSender._encode_contribution(contribution)
    decoded = ArrowHopInvocationContributionServer._decode_contribution(
        batch,
        row_index=0,
    )

    assert decoded.source_context == contribution.source_context
    assert decoded.destination_hop == contribution.destination_hop
    np.testing.assert_array_equal(
        decoded.bundle.bundle["input"].as_numpy(),
        contribution.bundle.bundle["input"].as_numpy(),
    )


class _Coordinator(HopInvocationContributionReceiveCoordinator):
    def __init__(self) -> None:
        self.received: HopInvocationContribution | None = None

    @override
    async def pass_hop_invocation_contribution(
        self,
        contribution: HopInvocationContribution,
    ) -> HopInvocationContributionAck:
        self.received = contribution
        return HopInvocationContributionAck(
            source_context=contribution.source_context,
            destination_hop=contribution.destination_hop,
        )


class _Directory:
    def __init__(self, worker: WorkerInstance) -> None:
        self.worker = worker

    async def get_worker_instance_by_id(self, worker_id):  # type: ignore[no-untyped-def]
        assert worker_id == self.worker.service_id
        return self.worker


@pytest.mark.asyncio
async def test_sender_reaches_generic_hop_server() -> None:
    worker_id = ServiceId()
    contribution = _contribution(worker_id)
    coordinator = _Coordinator()
    server = ArrowHopInvocationContributionServer(
        "grpc+tcp://127.0.0.1:0",
        coordinator,
        asyncio.get_running_loop(),
    )
    server_thread = Thread(target=server.serve, daemon=True)
    server_thread.start()
    worker = WorkerInstance(
        service_id=worker_id,
        capabilities=[
            Capability(
                type=CapabilityType.HOP_INVOCATION,
                interfaces=[
                    CapabilityInterface(
                        protocol=CapabilityProtocol.ARROW,
                        endpoint=ArrowUrl(
                            f"grpc+tcp://127.0.0.1:{server.port}"
                        ),
                    )
                ],
            ),
            Capability(type=CapabilityType.PLAN_ACTUATION, interfaces=[]),
            Capability(type=CapabilityType.NETWORK_PROFILING, interfaces=[]),
            Capability(type=CapabilityType.EXECUTION_PROFILING, interfaces=[]),
            Capability(type=CapabilityType.RESOURCE_PROFILING, interfaces=[]),
        ],
    )
    sender = ArrowHopInvocationContributionSender(
        _Directory(worker),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )

    try:
        await sender.send(contribution)
    finally:
        server.shutdown()
        server_thread.join(timeout=1)

    assert coordinator.received is not None
    assert coordinator.received.source_context == contribution.source_context
    assert coordinator.received.destination_hop == contribution.destination_hop
