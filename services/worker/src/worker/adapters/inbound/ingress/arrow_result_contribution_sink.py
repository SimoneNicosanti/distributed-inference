# mypy: disable-error-code="attr-defined,name-defined,misc"

import asyncio

import pyarrow as pa
from pyarrow import flight

from integration.ingress.arrow.result_sink.model_pass_result.arrow_model_pass_result_contribution_callback_schema import (
    CALLBACK_CONTRIBUTION_COMMAND,
    CONTEXT_FIELD_NAME,
    validate_contribution_schema,
)
from integration.dto.context.context_dto import ModelPassResultContributionMetadataDto
from shared.tensor.tensor import Tensor
from shared.tensor.tensor_bundle import TensorBundle
from worker.adapters.context_mapper import model_pass_context_from_dto
from worker.application.ports.inbound.ingress.result_contribution_sink import (
    ResultContributionSink,
)
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class ArrowResultContributionSink(flight.FlightServerBase):
    def __init__(
        self,
        location: str,
        contribution_sink: ResultContributionSink,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._contribution_sink = contribution_sink
        self._event_loop = event_loop
        super().__init__(location)

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Expected command descriptor")

        if descriptor.command != CALLBACK_CONTRIBUTION_COMMAND:
            raise pa.ArrowInvalid("Unsupported result contribution command")

        try:
            validate_contribution_schema(reader.schema)
        except ValueError as error:
            raise pa.ArrowInvalid(f"Invalid contribution schema: {error}") from error

        if not self._event_loop.is_running():
            raise flight.FlightUnavailableError("Ingress event loop is not running")

        received_batch: pa.RecordBatch | None = None

        for chunk in reader:
            batch = chunk.data

            if batch is None:
                continue

            if received_batch is not None:
                raise pa.ArrowInvalid("Expected exactly one batch per do_put")

            if batch.num_rows != 1:
                raise pa.ArrowInvalid("Expected exactly one contribution per batch")

            received_batch = batch

        if received_batch is None:
            raise pa.ArrowInvalid("Missing contribution batch")

        contribution = self._decode_result_contribution(
            received_batch,
            row_index=0,
        )

        future = asyncio.run_coroutine_threadsafe(
            self._contribution_sink.sink_contribution(contribution),
            self._event_loop,
        )

        try:
            future.result()
        except Exception as error:
            raise flight.FlightInternalError(
                "Result contribution handling failed"
            ) from error

    @staticmethod
    def _decode_result_contribution(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> ModelPassResultContribution:
        if row_index < 0 or row_index >= batch.num_rows:
            raise pa.ArrowInvalid(f"Invalid contribution row: {row_index}")

        try:
            batch.validate(full=True)
            validate_contribution_schema(batch.schema)
        except (pa.ArrowInvalid, ValueError) as error:
            raise pa.ArrowInvalid(f"Invalid contribution batch: {error}") from error

        context_index = batch.schema.get_field_index(CONTEXT_FIELD_NAME)
        metadata_scalar = batch.column(context_index)[row_index]

        if not metadata_scalar.is_valid:
            raise pa.ArrowInvalid("Missing contribution context")

        try:
            contribution_metadata = (
                ModelPassResultContributionMetadataDto.model_validate_json(
                    metadata_scalar.as_py()
                )
            )
        except ValueError as error:
            raise pa.ArrowInvalid(f"Invalid contribution context: {error}") from error

        bundle: dict[str, Tensor] = {}

        for column_index, field in enumerate(batch.schema):
            if field.name == CONTEXT_FIELD_NAME:
                continue

            column = batch.column(column_index)

            if not isinstance(column, pa.FixedShapeTensorArray):
                raise pa.ArrowInvalid(f"Field {field.name} is not a fixed-shape tensor")

            tensor_scalar = column[row_index]

            if not tensor_scalar.is_valid:
                raise pa.ArrowInvalid(f"Tensor {field.name} is null")

            if not isinstance(tensor_scalar, pa.FixedShapeTensorScalar):
                raise pa.ArrowInvalid(f"Invalid tensor scalar: {field.name}")

            bundle[field.name] = Tensor(value=tensor_scalar.to_numpy().copy())

        try:
            return ModelPassResultContribution(
                model_pass_context=model_pass_context_from_dto(
                    contribution_metadata.model_pass_context
                ),
                bundle=TensorBundle(bundle=bundle),
            )
        except ValueError as error:
            raise pa.ArrowInvalid(f"Invalid contribution context: {error}") from error
