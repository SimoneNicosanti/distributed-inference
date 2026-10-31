import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa

from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_retrieve_server import (
    ArrowModelPassResultRetrieveServer,
)
from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.application.sink.default_model_pass_result_retriever import (
    DefaultModelPassResultRetriever,
)
from worker.domain.context.model_invocation_context import ModelInvocationContext
from worker.domain.context.model_pass_context import ModelPassContext, ModelPassType
from worker.domain.model.model_pass import ModelPassResult


def _context() -> ModelPassContext:
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId()),
        ),
        plan_version=PlanVersion(version_number=1),
        model_pass_type=ModelPassType.FORWARD,
    )


def test_server_dispatches_to_retriever() -> None:
    result_retriever = Mock(spec=DefaultModelPassResultRetriever)
    server = ArrowModelPassResultRetrieveServer(
        "grpc+tcp://127.0.0.1:0", result_retriever, asyncio.get_running_loop()
    )
    call_arguments = (Mock(), Mock(), Mock(), Mock())

    try:
        server.do_exchange(*call_arguments)
    finally:
        server.shutdown()

    result_retriever.do_exchange.assert_called_once_with(*call_arguments)


class _ResultRetriever(ModelPassResultRetriever):
    def __init__(self, result: ModelPassResult) -> None:
        self.result = result
        self.requested_id = None

    async def retrieve_result(self, request_id):  # type: ignore[no-untyped-def]
        self.requested_id = request_id
        return self.result


class _Reader:
    def __init__(self, batch: pa.RecordBatch) -> None:
        self.schema = batch.schema
        self._batch = batch

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield SimpleNamespace(data=self._batch)


class _Writer:
    def __init__(self) -> None:
        self.schema = None
        self.batch = None

    def begin(self, schema: pa.Schema) -> None:
        self.schema = schema

    def write_batch(self, batch: pa.RecordBatch) -> None:
        self.batch = batch
