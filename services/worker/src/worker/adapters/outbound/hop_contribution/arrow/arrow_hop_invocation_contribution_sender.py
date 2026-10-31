# mypy: disable-error-code="attr-defined,name-defined"

import asyncio
from contextlib import closing
from typing import override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from integration.dto.context.context_dto import (
    HopInvocationContributionAckDto,
    HopInvocationContributionMetadataDto,
)
from integration.service_instance.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.service_instance.worker_instance import WorkerInstance
from integration.worker.hop_contribution.arrow.arrow_hop_invocation_contribution_schema import (
    SUBMIT_CONTRIBUTION_COMMAND,
    build_contribution_schema,
)
from shared.service.service import WorkerId
from worker.adapters.context_mapper import (
    destination_hop_to_dto,
    hop_invocation_context_from_dto,
    hop_invocation_context_to_dto,
)
from worker.application.ports.outbound.catalog.hop_deployment.hop_deployment_directory import (
    HopDeploymentDirectory,
)
from worker.application.ports.outbound.catalog.service.service_directory_gateway import (
    ServiceDirectoryGateway,
)
from worker.application.ports.outbound.hop_contribution_snd.contribution_sender import (
    ContributionSender,
)
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class ArrowHopInvocationContributionSender(ContributionSender):
    def __init__(
        self,
        directory: ServiceDirectoryGateway,
        hop_deployment_directory: HopDeploymentDirectory,
        timeout_s: float = 30.0,
    ) -> None:
        self._directory = directory
        self._hop_deployment_directory = hop_deployment_directory
        self._timeout_s = timeout_s

    @override
    async def send(self, contribution: HopInvocationContribution) -> None:
        worker_id = await self._resolve_worker_id(contribution)
        worker_instance = await self._directory.get_worker_instance_by_id(worker_id)
        interface = self._select_arrow_interface(worker_instance)

        await asyncio.to_thread(self._send_sync, interface, contribution)

    async def _resolve_worker_id(
        self, contribution: HopInvocationContribution
    ) -> WorkerId:
        deployment_id = await self._hop_deployment_directory.get_hop_deployment_id(
            plan_version=contribution.plan_version,
            replica_id=contribution.destination_hop,
        )
        return deployment_id.worker_id

    @staticmethod
    def _select_arrow_interface(
        worker_instance: WorkerInstance,
    ) -> CapabilityInterface:
        interfaces = worker_instance.get_capability_interfaces_by_type_and_protocol(
            CapabilityType.HOP_INVOCATION,
            CapabilityProtocol.ARROW,
        )
        if len(interfaces) != 1:
            raise ValueError(
                f"Worker {worker_instance.service_id} must expose exactly one "
                "Arrow hop invocation interface"
            )
        return interfaces[0]

    def _send_sync(
        self,
        interface: CapabilityInterface,
        contribution: HopInvocationContribution,
    ) -> None:
        batch = self._encode_contribution(contribution)
        descriptor = flight.FlightDescriptor.for_command(SUBMIT_CONTRIBUTION_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)

        with closing(flight.connect(interface.endpoint.unicode_string())) as client:
            writer, metadata_reader = client.do_put(
                descriptor,
                batch.schema,
                options=options,
            )
            with closing(writer):
                writer.write_batch(batch)
                writer.done_writing()
                ack_buffer = metadata_reader.read()
                extra_ack = metadata_reader.read()

        if ack_buffer is None:
            raise RuntimeError("Missing hop invocation acknowledgement")
        if extra_ack is not None:
            raise RuntimeError("Received multiple acknowledgements")

        ack_dto = HopInvocationContributionAckDto.model_validate_json(
            ack_buffer.to_pybytes()
        )
        ack_source_context = hop_invocation_context_from_dto(ack_dto.source_context)
        if ack_source_context != contribution.source_context:
            raise RuntimeError("Hop invocation acknowledgement source mismatch")
        if ack_dto.destination_hop != destination_hop_to_dto(
            contribution.destination_hop
        ):
            raise RuntimeError("Hop invocation acknowledgement destination mismatch")

    @staticmethod
    def _encode_contribution(
        contribution: HopInvocationContribution,
    ) -> pa.RecordBatch:
        tensor_values = {
            name: np.ascontiguousarray(tensor.as_numpy())
            for name, tensor in contribution.bundle.bundle.items()
        }
        schema = build_contribution_schema(tensor_values)
        metadata = HopInvocationContributionMetadataDto(
            source_context=hop_invocation_context_to_dto(contribution.source_context),
            destination_hop=destination_hop_to_dto(contribution.destination_hop),
        )

        arrays: list[pa.Array] = [
            pa.array([metadata.model_dump_json()], type=pa.large_string())
        ]
        arrays.extend(
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensor_values.values()
        )
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
