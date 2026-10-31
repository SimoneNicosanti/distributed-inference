# mypy: disable-error-code="attr-defined,name-defined"

import asyncio
from contextlib import closing
from typing import override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from front_end.application.ports.outbound.ingress_gateway import (
    IngressGateway,
)
from front_end.domain.inference_result import InferenceResult
from front_end.domain.tensor_map import TensorMap
from integration.shared.id_json import flow_id_to_json
from integration.worker.ingress.arrow.arrow_model_invocation_schema import (
    RUN_MODEL_INVOCATION_COMMAND,
    build_model_invocation_request_schema,
    validate_model_invocation_result_schema,
)
from shared.flow.flow import FlowId


class ArrowWorkerIngressGateway(IngressGateway):
    def __init__(
        self,
        location: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._location = location
        self._timeout_s = timeout_s

    @override
    async def invoke_model(
        self,
        flow_id: FlowId,
        bundle: TensorMap,
    ) -> InferenceResult:
        return await asyncio.to_thread(
            self._run_model_invocation_sync,
            flow_id,
            bundle,
        )

    def _run_model_invocation_sync(
        self,
        flow_id: FlowId,
        bundle: TensorMap,
    ) -> InferenceResult:
        request_batch = self._encode_request(flow_id, bundle)
        descriptor = flight.FlightDescriptor.for_command(RUN_MODEL_INVOCATION_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)

        with closing(flight.connect(self._location)) as client:
            writer, reader = client.do_exchange(descriptor, options=options)

            with closing(writer):
                writer.begin(request_batch.schema)
                writer.write_batch(request_batch)
                writer.done_writing()
                response_table = reader.read_all()

        if response_table.num_rows != 1:
            raise RuntimeError("Expected exactly one model invocation result")

        response_batches = response_table.combine_chunks().to_batches()
        if len(response_batches) != 1:
            raise RuntimeError("Expected one model invocation result batch")

        return self._decode_response(response_batches[0], row_index=0)

    @staticmethod
    def _encode_request(
        flow_id: FlowId,
        bundle: TensorMap,
    ) -> pa.RecordBatch:
        tensor_values = {
            name: np.ascontiguousarray(value) for name, value in bundle.items()
        }
        schema = build_model_invocation_request_schema(tensor_values)

        arrays: list[pa.Array] = [
            pa.array([flow_id_to_json(flow_id)], type=pa.large_string())
        ]
        arrays.extend(
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensor_values.values()
        )

        return pa.RecordBatch.from_arrays(arrays, schema=schema)

    @staticmethod
    def _decode_response(
        batch: pa.RecordBatch,
        row_index: int,
    ) -> InferenceResult:
        if row_index < 0 or row_index >= batch.num_rows:
            raise ValueError(f"Invalid model invocation result row: {row_index}")

        batch.validate(full=True)
        validate_model_invocation_result_schema(batch.schema)

        tensors: TensorMap = {}

        for column_index, field in enumerate(batch.schema):
            column = batch.column(column_index)
            if not isinstance(column, pa.FixedShapeTensorArray):
                raise ValueError(f"Field {field.name} is not a fixed-shape tensor")

            tensor_scalar = column[row_index]
            if not tensor_scalar.is_valid:
                raise ValueError(f"Tensor {field.name} is null")

            if not isinstance(tensor_scalar, pa.FixedShapeTensorScalar):
                raise ValueError(f"Invalid tensor scalar: {field.name}")

            tensors[field.name] = tensor_scalar.to_numpy().copy()

        return InferenceResult(bundle=tensors)
