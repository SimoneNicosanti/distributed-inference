# mypy: disable-error-code="attr-defined,name-defined"

import asyncio
from contextlib import closing
from typing import override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from catalog.api.service.service_directory import ServiceDirectory
from front_end.adapters.outbound.mappers.context_dto_mapper import (
    external_source_context_to_dto,
)
from front_end.adapters.outbound.worker_endpoint_resolver import (
    get_arrow_worker_endpoint,
)
from front_end.application.ports.outbound.inference.ingress_gateway import (
    IngressGateway,
)
from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.tensor_map import TensorMap
from integration.dto.context.context_dto import (
    HopInvocationContributionAckDto,
    HopInvocationContributionMetadataDto,
    IngressHopDto,
)
from integration.service_instance.service_instance import CapabilityType
from integration.worker.hop_contribution.arrow.arrow_hop_invocation_contribution_schema import (
    SUBMIT_CONTRIBUTION_COMMAND,
    build_contribution_schema,
)
from shared.plan.hop_logical_id import PlannedIngressReplicaId
from shared.service.service import WorkerId


class ArrowWorkerIngressGateway(IngressGateway):
    def __init__(self, directory: ServiceDirectory, timeout_s: float = 30.0) -> None:
        self._directory = directory
        self._timeout_s = timeout_s

    @override
    async def submit_model_pass(
        self,
        external_source_invocation_context: ExternalSourceInvocationContext,
        bundle: TensorMap,
        planned_ingress_replica_id: PlannedIngressReplicaId,
        worker_id: WorkerId,
    ) -> None:
        location = await get_arrow_worker_endpoint(
            self._directory,
            worker_id,
            CapabilityType.HOP_INVOCATION,
        )
        metadata = HopInvocationContributionMetadataDto(
            source_context=external_source_context_to_dto(
                external_source_invocation_context
            ),
            destination_hop=IngressHopDto(
                ingress_replica_id=planned_ingress_replica_id
            ),
        )
        await asyncio.to_thread(self._submit_sync, location, metadata, bundle)

    def _submit_sync(
        self,
        location: str,
        metadata: HopInvocationContributionMetadataDto,
        bundle: TensorMap,
    ) -> None:
        batch = self._encode_contribution(metadata, bundle)
        descriptor = flight.FlightDescriptor.for_command(SUBMIT_CONTRIBUTION_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)
        with closing(flight.connect(location)) as client:
            writer, reader = client.do_put(descriptor, batch.schema, options=options)
            with closing(writer):
                writer.write_batch(batch)
                writer.done_writing()
                ack_buffer = reader.read()
                extra_ack = reader.read()

        if ack_buffer is None:
            raise RuntimeError("Missing hop invocation acknowledgement")
        if extra_ack is not None:
            raise RuntimeError("Received multiple hop invocation acknowledgements")
        ack = HopInvocationContributionAckDto.model_validate_json(
            ack_buffer.to_pybytes()
        )
        if ack.source_context != metadata.source_context:
            raise RuntimeError("Hop invocation acknowledgement source mismatch")
        if ack.destination_hop != metadata.destination_hop:
            raise RuntimeError("Hop invocation acknowledgement destination mismatch")

    @staticmethod
    def _encode_contribution(
        metadata: HopInvocationContributionMetadataDto,
        bundle: TensorMap,
    ) -> pa.RecordBatch:
        tensors = {
            name: value if value.flags.c_contiguous else np.ascontiguousarray(value)
            for name, value in bundle.items()
        }
        schema = build_contribution_schema(tensors)
        arrays: list[pa.Array] = [
            pa.array([metadata.model_dump_json()], type=pa.large_string())
        ]
        arrays.extend(
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensors.values()
        )
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
