import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import flight

from front_end.adapters.outbound.inference.arrow_worker_sink_gateway import (
    ArrowWorkerSinkGateway,
)
from front_end.application.ports.outbound.inference.sink_gateway import (
    ModelPassFailedError,
)
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    CONTEXT_FIELD_NAME,
    build_result_payload_schema,
)
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId
from worker.adapters.context_mapper import model_pass_context_to_dto
from worker.adapters.inbound.sink.arrow.arrow_model_pass_outcome_retrieve_server import (
    ArrowModelPassOutcomeRetrieveServer,
)
from worker.domain.model.model_pass import ModelPassError, ModelPassResult
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


def _context():
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId())
        ),
        plan_version=PlanVersion(1),
        model_pass_type=ModelPassType.FORWARD,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_outcome_crosses_flight_transport(failure):
    context = _context()
    values = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    message = "Esecuzione fallita — input non valido"
    outcome = (
        ModelPassError(context, message)
        if failure
        else ModelPassResult(
            context, TensorBundle({"output": Tensor.from_numpy(values)})
        )
    )
    retriever = SimpleNamespace(retrieve_outcome=AsyncMock(return_value=outcome))
    server = ArrowModelPassOutcomeRetrieveServer(
        "grpc+tcp://127.0.0.1:0", retriever, asyncio.get_running_loop()
    )
    try:
        operation = asyncio.to_thread(
            ArrowWorkerSinkGateway(None, timeout_s=5)._retrieve_sync,
            f"grpc+tcp://127.0.0.1:{server.port}",
            model_pass_context_to_dto(context),
        )
        if failure:
            with pytest.raises(ModelPassFailedError) as caught:
                await operation
            assert str(caught.value) == message
        else:
            result = await operation
            np.testing.assert_array_equal(result["output"], values)
            assert result["output"].dtype == values.dtype
        retriever.retrieve_outcome.assert_awaited_once_with(context.model_pass_id)
    finally:
        await asyncio.to_thread(server.shutdown)


@pytest.mark.asyncio
async def test_retrieval_failure_remains_a_flight_error():
    context = _context()
    retriever = SimpleNamespace(
        retrieve_outcome=AsyncMock(side_effect=RuntimeError("storage failed"))
    )
    server = ArrowModelPassOutcomeRetrieveServer(
        "grpc+tcp://127.0.0.1:0", retriever, asyncio.get_running_loop()
    )
    try:
        with pytest.raises(flight.FlightInternalError, match="retrieval failed"):
            await asyncio.to_thread(
                ArrowWorkerSinkGateway(None, timeout_s=5)._retrieve_sync,
                f"grpc+tcp://127.0.0.1:{server.port}",
                model_pass_context_to_dto(context),
            )
    finally:
        await asyncio.to_thread(server.shutdown)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_status",
        "missing_status",
        "missing_message",
        "success_without_tensors",
        "failure_with_tensors",
        "null_metadata",
        "wrong_context",
    ],
)
def test_rejects_invalid_outcomes(change):
    context = _context()
    batch = ArrowModelPassOutcomeRetrieveServer._encode_outcome(
        ModelPassError(context, "failed")
    )
    metadata = json.loads(batch.column(0)[0].as_py())
    schema = batch.schema
    if change == "unknown_status":
        metadata["status"] = "unknown"
    elif change == "missing_status":
        del metadata["status"]
    elif change == "missing_message":
        del metadata["message"]
    elif change == "success_without_tensors":
        metadata = {
            "status": "success",
            "model_pass_context": metadata["model_pass_context"],
        }
    elif change == "failure_with_tensors":
        schema = build_result_payload_schema({"output": np.ones(1, dtype=np.float32)})
    elif change == "wrong_context":
        metadata["model_pass_context"]["model_pass_id"] = str(uuid4())
    arrays = [
        pa.array(
            [None if change == "null_metadata" else json.dumps(metadata)],
            type=pa.large_string(),
        )
    ]
    if change == "failure_with_tensors":
        arrays.append(
            pa.FixedShapeTensorArray.from_numpy_ndarray(
                np.ones((1, 1), dtype=np.float32)
            )
        )
    invalid = pa.RecordBatch.from_arrays(arrays, schema=schema)
    with pytest.raises((ValueError, pa.ArrowInvalid)):
        ArrowWorkerSinkGateway._decode_result(
            invalid, model_pass_context_to_dto(context)
        )


def test_failure_has_no_tensors_and_allows_empty_message():
    context = _context()
    batch = ArrowModelPassOutcomeRetrieveServer._encode_outcome(
        ModelPassError(context, "")
    )
    assert batch.schema.names == [CONTEXT_FIELD_NAME]
    with pytest.raises(ModelPassFailedError) as caught:
        ArrowWorkerSinkGateway._decode_result(batch, model_pass_context_to_dto(context))
    assert str(caught.value) == ""
