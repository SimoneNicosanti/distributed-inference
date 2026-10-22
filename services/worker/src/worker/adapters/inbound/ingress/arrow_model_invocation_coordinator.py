# mypy: disable-error-code="attr-defined,name-defined"

import asyncio

import numpy as np
import pyarrow as pa
from pyarrow import flight

from integration.shared.id_json import flow_id_from_json
from integration.worker.ingress.arrow.arrow_model_invocation_schema import (
    FLOW_ID_FIELD_NAME,
    RUN_MODEL_INVOCATION_COMMAND,
    build_model_invocation_result_schema,
    validate_model_invocation_request_schema,
)
from shared.flow.flow import FlowId
from shared.tensor.tensor import Tensor
from shared.tensor.tensor_bundle import TensorBundle
from worker.application.ports.inbound.ingress.model_invocation_coordinator import (
    ModelInvocationCoordinator,
)


class ArrowModelInvocationCoordinator(
    flight.FlightServerBase  # type: ignore[misc]
):
    def __init__(
        self,
        location: str,
        coordinator: ModelInvocationCoordinator,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._coordinator = coordinator
        self._event_loop = event_loop
        super().__init__(location)

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightStreamWriter,
    ) -> None:
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Expected command descriptor")

        if descriptor.command != RUN_MODEL_INVOCATION_COMMAND:
            raise pa.ArrowInvalid("Unsupported model invocation command")

        try:
            validate_model_invocation_request_schema(reader.schema)
        except ValueError as error:
            raise pa.ArrowInvalid(
                f"Invalid model invocation schema: {error}"
            ) from error

        if not self._event_loop.is_running():
            raise flight.FlightUnavailableError("Ingress event loop is not running")

        received_batch: pa.RecordBatch | None = None

        for chunk in reader:
            batch = chunk.data

            if batch is None:
                continue

            if received_batch is not None:
                raise pa.ArrowInvalid("Expected exactly one batch per do_exchange")

            if batch.num_rows != 1:
                raise pa.ArrowInvalid("Expected exactly one model invocation per batch")

            received_batch = batch

        if received_batch is None:
            raise pa.ArrowInvalid("Missing model invocation batch")

        flow_id, bundle = self._decode_request(received_batch, row_index=0)
        future = asyncio.run_coroutine_threadsafe(
            self._coordinator.run_model_invocation(flow_id, bundle),
            self._event_loop,
        )

        try:
            result = future.result()
        except Exception as error:
            raise flight.FlightInternalError("Model invocation failed") from error

        response_batch = self._encode_result(result.bundle)
        writer.begin(response_batch.schema)
        writer.write_batch(response_batch)

    @staticmethod
    def _decode_request(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> tuple[FlowId, TensorBundle]:
        if row_index < 0 or row_index >= batch.num_rows:
            raise pa.ArrowInvalid(f"Invalid model invocation row: {row_index}")

        try:
            batch.validate(full=True)
            validate_model_invocation_request_schema(batch.schema)
        except (pa.ArrowInvalid, ValueError) as error:
            raise pa.ArrowInvalid(f"Invalid model invocation batch: {error}") from error

        flow_id_index = batch.schema.get_field_index(FLOW_ID_FIELD_NAME)
        flow_id_scalar = batch.column(flow_id_index)[row_index]

        if not flow_id_scalar.is_valid:
            raise pa.ArrowInvalid("Missing model invocation flow id")

        try:
            flow_id = flow_id_from_json(flow_id_scalar.as_py())
        except (TypeError, ValueError) as error:
            raise pa.ArrowInvalid(
                f"Invalid model invocation flow id: {error}"
            ) from error

        tensors: dict[str, Tensor] = {}

        for column_index, field in enumerate(batch.schema):
            if field.name == FLOW_ID_FIELD_NAME:
                continue

            column = batch.column(column_index)
            if not isinstance(column, pa.FixedShapeTensorArray):
                raise pa.ArrowInvalid(f"Field {field.name} is not a fixed-shape tensor")

            tensor_scalar = column[row_index]
            if not tensor_scalar.is_valid:
                raise pa.ArrowInvalid(f"Tensor {field.name} is null")

            if not isinstance(tensor_scalar, pa.FixedShapeTensorScalar):
                raise pa.ArrowInvalid(f"Invalid tensor scalar: {field.name}")

            tensors[field.name] = Tensor(value=tensor_scalar.to_numpy().copy())

        return flow_id, TensorBundle(bundle=tensors)

    @staticmethod
    def _encode_result(bundle: TensorBundle) -> pa.RecordBatch:
        tensor_values = {
            name: np.ascontiguousarray(tensor.value)
            for name, tensor in bundle.bundle.items()
        }
        schema = build_model_invocation_result_schema(tensor_values)

        arrays: list[pa.Array] = [
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensor_values.values()
        ]

        return pa.RecordBatch.from_arrays(arrays, schema=schema)
