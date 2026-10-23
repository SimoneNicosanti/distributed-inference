# mypy: disable-error-code="attr-defined,name-defined,misc"

import asyncio

import pyarrow as pa
from pyarrow import flight

from integration.dto.context.context_dto import (
    PartitionInvocationContributionAckDto,
    PartitionInvocationContributionMetadataDto,
)
from integration.worker.partition_invocation.arrow.arrow_partition_invocation_coordinator_schema import (
    CONTEXT_FIELD_NAME,
    SUBMIT_CONTRIBUTION_COMMAND,
    validate_contribution_schema,
)
from worker.adapters.context_mapper import (
    source_context_from_dto,
    source_context_to_dto,
)
from worker.application.ports.inbound.partition_invocation.partition_invocation_coordinator import (
    PartitionInvocationCoordinator,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


class ArrowPartitionInvocationCoordinator(flight.FlightServerBase):
    def __init__(
        self,
        location: str,
        coordinator: PartitionInvocationCoordinator,
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

            # Chunk contenente solo application metadata.
            if batch is None:
                continue

            if received_batch is not None:
                raise pa.ArrowInvalid("Expected exactly one batch per do_put")

            if batch.num_rows != 1:
                raise pa.ArrowInvalid("Expected exactly one contribution per batch")

            received_batch = batch

        if received_batch is None:
            raise pa.ArrowInvalid("Missing contribution batch")

        contribution = self._decode_invocation_contribution(
            received_batch,
            row_index=0,
        )

        future = asyncio.run_coroutine_threadsafe(
            self._coordinator.process_partition_invocation_contribution(contribution),
            self._event_loop,
        )

        try:
            ack = future.result()
        except Exception as error:
            raise flight.FlightInternalError("Partition invocation failed") from error

        ack_dto = PartitionInvocationContributionAckDto(
            context=source_context_to_dto(ack.context)
        )
        writer.write(pa.py_buffer(ack_dto.model_dump_json().encode("utf-8")))

    @staticmethod
    def _decode_invocation_contribution(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> PartitionInvocationContribution:
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
                PartitionInvocationContributionMetadataDto.model_validate_json(
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

            if not isinstance(
                column,
                pa.FixedShapeTensorArray,
            ):
                raise pa.ArrowInvalid(f"Field {field.name} is not a fixed-shape tensor")

            tensor_scalar = column[row_index]

            if not tensor_scalar.is_valid:
                raise pa.ArrowInvalid(f"Tensor {field.name} is null")

            if not isinstance(
                tensor_scalar,
                pa.FixedShapeTensorScalar,
            ):
                raise pa.ArrowInvalid(f"Invalid tensor scalar: {field.name}")

            value = tensor_scalar.to_numpy().copy()

            bundle[field.name] = Tensor.from_numpy(value)

        try:
            return PartitionInvocationContribution(
                source_invocation_context=source_context_from_dto(
                    contribution_metadata.source_invocation_context
                ),
                target_replica_id=contribution_metadata.target_replica_id,
                bundle=TensorBundle(bundle=bundle),
            )
        except ValueError as error:
            raise pa.ArrowInvalid(f"Invalid contribution context: {error}") from error
