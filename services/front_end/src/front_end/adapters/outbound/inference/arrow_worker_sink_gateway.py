# mypy: disable-error-code="attr-defined,name-defined"

import asyncio
from contextlib import closing
from typing import override

import pyarrow as pa
from pyarrow import flight

from catalog.api.service.service_directory import ServiceDirectory
from front_end.adapters.outbound.mappers.context_dto_mapper import (
    model_pass_context_to_dto,
)
from front_end.adapters.outbound.worker_endpoint_resolver import (
    get_arrow_worker_endpoint,
)
from front_end.application.ports.outbound.inference.sink_gateway import SinkGateway
from front_end.domain.tensor_map import TensorMap
from integration.dto.context.context_dto import (
    ModelPassContextDto,
    ModelPassResultMetadataDto,
)
from integration.service_instance.service_instance import CapabilityType
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    CONTEXT_FIELD_NAME,
    RETRIEVE_RESULT_COMMAND,
    build_retrieve_request_schema,
    validate_result_payload_schema,
)
from shared.context.model_pass_context import ModelPassContext
from shared.service.service import WorkerId


class ArrowWorkerSinkGateway(SinkGateway):
    def __init__(self, directory: ServiceDirectory, timeout_s: float = 300.0) -> None:
        self._directory = directory
        self._timeout_s = timeout_s

    @override
    async def retrieve_result(
        self,
        model_pass_context: ModelPassContext,
        worker_id: WorkerId,
    ) -> TensorMap:
        location = await get_arrow_worker_endpoint(
            self._directory,
            worker_id,
            CapabilityType.RESULT_RETRIEVE,
        )
        return await asyncio.to_thread(
            self._retrieve_sync, location, model_pass_context_to_dto(model_pass_context)
        )

    def _retrieve_sync(self, location: str, context: ModelPassContextDto) -> TensorMap:
        schema = build_retrieve_request_schema()
        batch = pa.RecordBatch.from_arrays(
            [pa.array([str(context.model_pass_id)], type=pa.large_string())],
            schema=schema,
        )
        descriptor = flight.FlightDescriptor.for_command(RETRIEVE_RESULT_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)
        with closing(flight.connect(location)) as client:
            writer, reader = client.do_exchange(descriptor, options=options)
            with closing(writer):
                writer.begin(schema)
                writer.write_batch(batch)
                writer.done_writing()
                response = reader.read_all()

        if response.num_rows != 1:
            raise RuntimeError("Expected exactly one model pass result")
        batches = response.combine_chunks().to_batches()
        if len(batches) != 1:
            raise RuntimeError("Expected exactly one model pass result batch")
        return self._decode_result(batches[0], context)

    @staticmethod
    def _decode_result(
        batch: pa.RecordBatch, context: ModelPassContextDto
    ) -> TensorMap:
        batch.validate(full=True)
        validate_result_payload_schema(batch.schema)
        if batch.num_rows != 1:
            raise ValueError("Expected exactly one model pass result row")
        metadata_value = batch.column(batch.schema.get_field_index(CONTEXT_FIELD_NAME))[
            0
        ]
        if not metadata_value.is_valid:
            raise ValueError("Missing model pass result context")
        metadata = ModelPassResultMetadataDto.model_validate_json(
            metadata_value.as_py()
        )
        if metadata.model_pass_context != context:
            raise ValueError("Model pass result context does not match request")

        tensors: TensorMap = {}
        for column_index, field in enumerate(batch.schema):
            if field.name == CONTEXT_FIELD_NAME:
                continue
            column = batch.column(column_index)
            if not isinstance(column, pa.FixedShapeTensorArray):
                raise ValueError(f"Field {field.name} is not a fixed-shape tensor")
            scalar = column[0]
            if not scalar.is_valid or not isinstance(scalar, pa.FixedShapeTensorScalar):
                raise ValueError(f"Invalid result tensor: {field.name}")
            tensors[field.name] = scalar.to_numpy().copy()
        return tensors
