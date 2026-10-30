# mypy: disable-error-code="attr-defined,name-defined,misc"

import asyncio

import pyarrow as pa
from pyarrow import flight

from integration.dto.context.context_dto import (
    HopInvocationContributionAckDto,
    HopInvocationContributionMetadataDto,
)
from integration.worker.hop_contribution.arrow.arrow_hop_invocation_contribution_schema import (
    CONTEXT_FIELD_NAME,
    SUBMIT_CONTRIBUTION_COMMAND,
    validate_contribution_schema,
)
from worker.adapters.context_mapper import (
    destination_hop_from_dto,
    destination_hop_to_dto,
    hop_invocation_context_from_dto,
    hop_invocation_context_to_dto,
)
from worker.application.ports.inbound.hop_contribution_rcv.hop_invocation_contribution_receive_coordinator import (
    HopInvocationContributionReceiveCoordinator,
)
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


class ArrowHopInvocationContributionServer(flight.FlightServerBase):
    def __init__(
        self,
        location: str,
        coordinator: HopInvocationContributionReceiveCoordinator,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._coordinator = coordinator
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
        if descriptor.command != SUBMIT_CONTRIBUTION_COMMAND:
            raise pa.ArrowInvalid("Unsupported submit command")

        try:
            validate_contribution_schema(reader.schema)
        except ValueError as error:
            raise pa.ArrowInvalid(f"Invalid contribution schema: {error}") from error

        if not self._event_loop.is_running():
            raise flight.FlightUnavailableError("Worker event loop is not running")

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

        contribution = self._decode_contribution(received_batch, row_index=0)
        future = asyncio.run_coroutine_threadsafe(
            self._coordinator.pass_hop_invocation_contribution(contribution),
            self._event_loop,
        )

        try:
            ack = future.result()
        except Exception as error:
            raise flight.FlightInternalError("Hop invocation failed") from error

        ack_dto = HopInvocationContributionAckDto(
            source_context=hop_invocation_context_to_dto(ack.source_context),
            destination_hop=destination_hop_to_dto(ack.destination_hop),
        )
        writer.write(pa.py_buffer(ack_dto.model_dump_json().encode("utf-8")))

    @staticmethod
    def _decode_contribution(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> HopInvocationContribution:
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
            metadata = HopInvocationContributionMetadataDto.model_validate_json(
                metadata_scalar.as_py()
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

            bundle[field.name] = Tensor.from_numpy(tensor_scalar.to_numpy().copy())

        return HopInvocationContribution(
            source_context=hop_invocation_context_from_dto(metadata.source_context),
            destination_hop=destination_hop_from_dto(metadata.destination_hop),
            bundle=TensorBundle(bundle=bundle),
        )
