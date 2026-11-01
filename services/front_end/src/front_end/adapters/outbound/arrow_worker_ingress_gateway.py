# mypy: disable-error-code="attr-defined,name-defined"

from contextlib import closing
from typing import Any, override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from front_end.application.ports.outbound.ingress_gateway import (
    IngressGateway,
)
from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.inference_result import InferenceResult
from front_end.domain.tensor_map import TensorMap
from integration.worker.ingress.arrow.arrow_model_invocation_schema import (
    RUN_MODEL_INVOCATION_COMMAND,
    validate_model_invocation_result_schema,
)
from shared.flow.flow import FlowId
from shared.plan.hop_deployment_id import HopDeploymentId
from shared.plan.hop_logical_id import IngressReplicaId


class ArrowWorkerIngressGateway(IngressGateway):
    def __init__(
        self,
        location: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._location = location
        self._timeout_s = timeout_s

    @override
    async def submit_model_pass(
        self,
        external_source_invocation_context: ExternalSourceInvocationContext,
        bundle: TensorMap[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]],
        ingress_deployment_id: HopDeploymentId[IngressReplicaId],
    ) -> None:
        return await super().submit_model_pass(
            external_source_invocation_context, bundle, ingress_deployment_id
        )

    @override
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
        model_invocation_request: ModelInvocationRequest,
    ) -> pa.RecordBatch:
        validate_model_invocation_request_schema(model_invocation_request.schema)

        batch = pa.RecordBatch.from_arrays([[model_invocation_request.model_name]])
        batch.validate(full=True)

        return batch

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
