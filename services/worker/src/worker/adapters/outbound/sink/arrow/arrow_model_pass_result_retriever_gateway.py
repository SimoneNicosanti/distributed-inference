# mypy: disable-error-code="attr-defined,name-defined"

import asyncio
from contextlib import closing
from typing import override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from integration.dto.context.context_dto import ModelPassResultMetadataDto
from integration.service_instance.ingress_instance import ResultSinkInstance
from integration.service_instance.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    CONTEXT_FIELD_NAME,
    RETRIEVE_RESULT_COMMAND,
    build_retrieve_request_schema,
    validate_result_payload_schema,
)
from worker.domain.context.model_pass_context import ModelPassId
from shared.service.service import SinkId
from worker.adapters.context_mapper import model_pass_context_from_dto
from worker.application.ports.outbound.catalog.service.service_directory_gateway import (
    ServiceDirectoryGateway,
)
from worker.application.ports.outbound.sink.model_pass_result_gateway import (
    ModelPassResultRetrieverGateway,
)
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


class ArrowModelPassResultRetrieverGateway(ModelPassResultRetrieverGateway):
    def __init__(
        self,
        directory: ServiceDirectoryGateway,
        timeout_s: float = 30.0,
    ) -> None:
        self._directory = directory
        self._timeout_s = timeout_s

    @override
    async def retrieve_result(
        self,
        request_id: ModelPassId,
        result_sink_id: SinkId,
    ) -> ModelPassResult:
        result_sink = await self._directory.get_result_sink_instance_by_id(
            result_sink_id
        )
        interface = self._select_arrow_interface(result_sink)

        return await asyncio.to_thread(
            self._retrieve_sync,
            interface,
            request_id,
        )

    @staticmethod
    def _select_arrow_interface(
        result_sink: ResultSinkInstance,
    ) -> CapabilityInterface:
        interfaces = result_sink.get_capability_interfaces_by_type_and_protocol(
            CapabilityType.RESULT_SINKER,
            CapabilityProtocol.ARROW,
        )
        if not interfaces:
            raise ValueError(
                f"Result sink {result_sink.service_id} "
                "does not expose an Arrow result interface"
            )
        return interfaces[0]

    def _retrieve_sync(
        self,
        interface: CapabilityInterface,
        request_id: ModelPassId,
    ) -> ModelPassResult:
        request_batch = self._encode_request(request_id)
        descriptor = flight.FlightDescriptor.for_command(RETRIEVE_RESULT_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)

        with closing(flight.connect(interface.endpoint.unicode_string())) as client:
            writer, reader = client.do_exchange(descriptor, options=options)
            with closing(writer):
                writer.begin(request_batch.schema)
                writer.write_batch(request_batch)
                writer.done_writing()
                response_table = reader.read_all()

        if response_table.num_rows != 1:
            raise RuntimeError("Expected exactly one model pass result")

        response_batches = response_table.combine_chunks().to_batches()
        if len(response_batches) != 1:
            raise RuntimeError("Expected one model pass result batch")

        result = self._decode_response(response_batches[0], row_index=0)
        if result.context.model_pass_id != request_id:
            raise RuntimeError("Received a result for a different model pass")
        return result

    @staticmethod
    def _encode_request(request_id: ModelPassId) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([str(request_id)], type=pa.large_string())],
            schema=build_retrieve_request_schema(),
        )

    @staticmethod
    def _decode_response(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> ModelPassResult:
        if row_index < 0 or row_index >= batch.num_rows:
            raise ValueError(f"Invalid model pass result row: {row_index}")

        try:
            batch.validate(full=True)
            validate_result_payload_schema(batch.schema)
        except (pa.ArrowInvalid, ValueError) as error:
            raise ValueError(f"Invalid model pass result batch: {error}") from error

        context_index = batch.schema.get_field_index(CONTEXT_FIELD_NAME)
        context_scalar = batch.column(context_index)[row_index]
        if not context_scalar.is_valid:
            raise ValueError("Missing model pass result context")

        try:
            metadata = ModelPassResultMetadataDto.model_validate_json(
                context_scalar.as_py()
            )
        except ValueError as error:
            raise ValueError(f"Invalid model pass result context: {error}") from error

        tensors: dict[str, Tensor] = {}
        for column_index, field in enumerate(batch.schema):
            if field.name == CONTEXT_FIELD_NAME:
                continue

            column = batch.column(column_index)
            if not isinstance(column, pa.FixedShapeTensorArray):
                raise ValueError(f"Field {field.name} is not a fixed-shape tensor")

            tensor_scalar = column[row_index]
            if not tensor_scalar.is_valid:
                raise ValueError(f"Tensor {field.name} is null")
            if not isinstance(tensor_scalar, pa.FixedShapeTensorScalar):
                raise ValueError(f"Invalid tensor scalar: {field.name}")

            value = np.array(tensor_scalar.to_numpy(), copy=True)
            tensors[field.name] = Tensor.from_numpy(value)

        return ModelPassResult(
            context=model_pass_context_from_dto(metadata.model_pass_context),
            bundle=TensorBundle(bundle=tensors),
        )
