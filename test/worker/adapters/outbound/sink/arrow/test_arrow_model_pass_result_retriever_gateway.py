import asyncio
from threading import Thread
from unittest.mock import Mock

import numpy as np
import pytest

from integration.service_instance.ingress_instance import ResultSinkInstance
from integration.service_instance.service_instance import (
    ArrowUrl,
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_contribution_sink import (
    ArrowModelPassResultContributionSink,
)
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_retriever import (
    ArrowModelPassResultRetriever,
)
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_server import (
    ArrowModelPassResultServer,
)
from worker.adapters.outbound.sink.arrow.arrow_model_pass_result_retriever_gateway import (
    ArrowModelPassResultRetrieverGateway,
)
from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


class _Retriever(ModelPassResultRetriever):
    def __init__(self, result: ModelPassResult) -> None:
        self._result = result
        self.requested_id = None

    async def retrieve_result(self, request_id):  # type: ignore[no-untyped-def]
        self.requested_id = request_id
        return self._result


class _Directory:
    def __init__(self, instance: ResultSinkInstance) -> None:
        self._instance = instance
        self.requested_id = None

    async def get_result_sink_instance_by_id(self, result_sink_id):  # type: ignore[no-untyped-def]
        self.requested_id = result_sink_id
        return self._instance


@pytest.mark.asyncio
async def test_retrieves_result_from_sink_selected_by_id() -> None:
    sink_id = ServiceId()
    context = ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId()),
        ),
        plan_version=PlanVersion(version_number=1),
        model_pass_type=ModelPassType.FORWARD,
    )
    expected = ModelPassResult(
        context=context,
        bundle=TensorBundle(
            {"output": Tensor.from_numpy(np.array([1, 2], dtype=np.float32))}
        ),
    )
    application_retriever = _Retriever(expected)
    server = ArrowModelPassResultServer(
        "grpc+tcp://127.0.0.1:0",
        Mock(spec=ArrowModelPassResultContributionSink),
        ArrowModelPassResultRetriever(
            application_retriever,
            asyncio.get_running_loop(),
        ),
    )
    server_thread = Thread(target=server.serve, daemon=True)
    server_thread.start()
    directory = _Directory(
        ResultSinkInstance(
            service_id=sink_id,
            capabilities=[
                Capability(
                    type=CapabilityType.RESULT_SINKER,
                    interfaces=[
                        CapabilityInterface(
                            protocol=CapabilityProtocol.ARROW,
                            endpoint=ArrowUrl(f"grpc+tcp://127.0.0.1:{server.port}"),
                        )
                    ],
                )
            ],
        )
    )
    gateway = ArrowModelPassResultRetrieverGateway(directory)  # type: ignore[arg-type]

    try:
        actual = await gateway.retrieve_result(context.model_pass_id, sink_id)
    finally:
        server.shutdown()
        server_thread.join(timeout=1)

    assert directory.requested_id == sink_id
    assert application_retriever.requested_id == context.model_pass_id
    assert actual.context == expected.context
    np.testing.assert_array_equal(
        actual.bundle.bundle["output"].as_numpy(),
        expected.bundle.bundle["output"].as_numpy(),
    )
