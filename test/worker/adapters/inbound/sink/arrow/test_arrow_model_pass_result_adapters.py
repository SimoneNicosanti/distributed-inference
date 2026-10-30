import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import flight

from integration.dto.context.context_dto import ModelPassResultMetadataDto
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    CONTEXT_FIELD_NAME,
    RETRIEVE_RESULT_COMMAND,
    build_retrieve_request_schema,
    validate_result_payload_schema,
)
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId
from worker.adapters.context_mapper import model_pass_context_from_dto
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_retriever import (
    ArrowModelPassResultRetriever,
)
from worker.adapters.inbound.sink.arrow.arrow_model_pass_result_server import (
    ArrowModelPassResultServer,
)
from worker.application.ports.inbound.sink.model_pass_result_retriever import (
    ModelPassResultRetriever,
)
from worker.domain.context.model_invocation_context import ModelInvocationContext
from worker.domain.context.model_pass_context import ModelPassContext, ModelPassType
from worker.domain.model.model_pass import ModelPassResult
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


def _context() -> ModelPassContext:
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId()),
        ),
        plan_version=PlanVersion(version_number=1),
        model_pass_type=ModelPassType.FORWARD,
    )


def test_server_dispatches_to_retriever() -> None:
    result_retriever = Mock(spec=ArrowModelPassResultRetriever)
    server = ArrowModelPassResultServer(
        "grpc+tcp://127.0.0.1:0",
        result_retriever,
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


@pytest.mark.asyncio
async def test_retrieves_and_encodes_model_pass_result() -> None:
    context = _context()
    result = ModelPassResult(
        context=context,
        bundle=TensorBundle(
            {"output": Tensor.from_numpy(np.array([3], dtype=np.int64))}
        ),
    )
    application_retriever = _ResultRetriever(result)
    adapter = ArrowModelPassResultRetriever(
        application_retriever,
        asyncio.get_running_loop(),
    )
    request = pa.RecordBatch.from_arrays(
        [pa.array([str(context.model_pass_id)], type=pa.large_string())],
        schema=build_retrieve_request_schema(),
    )
    writer = _Writer()

    await asyncio.to_thread(
        adapter.do_exchange,
        None,
        flight.FlightDescriptor.for_command(RETRIEVE_RESULT_COMMAND),
        _Reader(request),
        writer,
    )

    assert application_retriever.requested_id == context.model_pass_id
    assert writer.batch is not None
    validate_result_payload_schema(writer.batch.schema)
    metadata = ModelPassResultMetadataDto.model_validate_json(
        writer.batch.column(CONTEXT_FIELD_NAME)[0].as_py()
    )
    assert model_pass_context_from_dto(metadata.model_pass_context) == context
    np.testing.assert_array_equal(
        writer.batch.column("output")[0].to_numpy(),
        np.array([3], dtype=np.int64),
    )
