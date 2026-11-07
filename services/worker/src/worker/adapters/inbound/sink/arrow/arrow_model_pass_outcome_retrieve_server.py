# mypy: disable-error-code="attr-defined,name-defined,misc"

import asyncio

# mypy: disable-error-code="attr-defined,name-defined"
from uuid import UUID

import numpy as np
import pyarrow as pa
from pyarrow import flight

from integration.dto.context.context_dto import (
    ModelPassErrorMetadataDto,
    ModelPassResultMetadataDto,
)
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    MODEL_PASS_ID_FIELD_NAME,
    RETRIEVE_RESULT_COMMAND,
    build_failure_payload_schema,
    build_result_payload_schema,
    validate_retrieve_request_schema,
)
from shared.context.model_pass_context import ModelPassId
from worker.adapters.context_mapper import model_pass_context_to_dto
from worker.application.ports.inbound.sink.model_pass_outcome_retriever import (
    ModelPassOutcomeRetriever,
)
from worker.domain.model.model_pass import ModelPassError, ModelPassOutcome


class ArrowModelPassOutcomeRetrieveServer(flight.FlightServerBase):
    def __init__(
        self,
        location: str,
        outcome_retriever: ModelPassOutcomeRetriever,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(location)
        self._result_retriever = outcome_retriever
        self._event_loop = event_loop

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightStreamWriter,
    ) -> None:
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Expected command descriptor")

        if descriptor.command != RETRIEVE_RESULT_COMMAND:
            raise pa.ArrowInvalid("Unsupported model pass result retrieval command")

        try:
            validate_retrieve_request_schema(reader.schema)
        except ValueError as error:
            raise pa.ArrowInvalid(
                f"Invalid result retrieval schema: {error}"
            ) from error

        if not self._event_loop.is_running():
            raise flight.FlightUnavailableError("Sink event loop is not running")

        received_batch: pa.RecordBatch | None = None

        for chunk in reader:
            batch = chunk.data

            if batch is None:
                continue

            if received_batch is not None:
                raise pa.ArrowInvalid("Expected exactly one batch per do_exchange")

            if batch.num_rows != 1:
                raise pa.ArrowInvalid("Expected exactly one model pass id per batch")

            received_batch = batch

        if received_batch is None:
            raise pa.ArrowInvalid("Missing model pass result retrieval batch")

        model_pass_id = self._decode_retrieve_request(received_batch, row_index=0)
        future = asyncio.run_coroutine_threadsafe(
            self._result_retriever.retrieve_outcome(model_pass_id),
            self._event_loop,
        )

        try:
            outcome = future.result()
        except Exception as error:
            raise flight.FlightInternalError(
                "Model pass result retrieval failed"
            ) from error

        response_batch = self._encode_outcome(outcome)
        writer.begin(response_batch.schema)
        writer.write_batch(response_batch)

    @staticmethod
    def _decode_retrieve_request(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> ModelPassId:
        if row_index < 0 or row_index >= batch.num_rows:
            raise pa.ArrowInvalid(f"Invalid result retrieval row: {row_index}")

        try:
            batch.validate(full=True)
            validate_retrieve_request_schema(batch.schema)
        except (pa.ArrowInvalid, ValueError) as error:
            raise pa.ArrowInvalid(f"Invalid result retrieval batch: {error}") from error

        model_pass_id_index = batch.schema.get_field_index(MODEL_PASS_ID_FIELD_NAME)
        model_pass_id_scalar = batch.column(model_pass_id_index)[row_index]

        if not model_pass_id_scalar.is_valid:
            raise pa.ArrowInvalid("Missing model pass id")

        try:
            return UUID(model_pass_id_scalar.as_py())
        except (AttributeError, TypeError, ValueError) as error:
            raise pa.ArrowInvalid(f"Invalid model pass id: {error}") from error

    @staticmethod
    def _encode_outcome(outcome: ModelPassOutcome) -> pa.RecordBatch:
        if isinstance(outcome, ModelPassError):
            failure_metadata = ModelPassErrorMetadataDto(
                model_pass_context=model_pass_context_to_dto(outcome.context),
                message=outcome.message,
            )
            return pa.RecordBatch.from_arrays(
                [pa.array([failure_metadata.model_dump_json()], type=pa.large_string())],
                schema=build_failure_payload_schema(),
            )

        tensor_values = {
            name: np.ascontiguousarray(tensor.as_numpy())
            for name, tensor in outcome.bundle.bundle.items()
        }
        schema = build_result_payload_schema(tensor_values)
        metadata = ModelPassResultMetadataDto(
            model_pass_context=model_pass_context_to_dto(outcome.context)
        )

        arrays: list[pa.Array] = [
            pa.array([metadata.model_dump_json()], type=pa.large_string())
        ]
        arrays.extend(
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensor_values.values()
        )

        return pa.RecordBatch.from_arrays(arrays, schema=schema)
