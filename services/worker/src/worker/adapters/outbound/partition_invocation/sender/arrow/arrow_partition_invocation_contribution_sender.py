import asyncio
from contextlib import closing
from typing import override

import numpy as np
import pyarrow as pa
from pyarrow import flight

from directory.contracts.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.worker.partition_invocation.arrow.arrow_partition_invocation_coordinator_schema import (
    SUBMIT_CONTRIBUTION_COMMAND,
    build_contribution_schema,
)
from worker.application.ports.outbound.directory.service_directory import (
    ServiceDirectory,
)
from worker.application.ports.outbound.partition_invocation.sender.partition_invocation_contribution_sender import (
    PartitionInvocationContributionSender,
)
from worker.domain.directory.worker_instance import WorkerInstance
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
    PartitionInvocationContributionAck,
)


class ArrowPartitionInvocationContributionSender(PartitionInvocationContributionSender):
    def __init__(
        self,
        directory: ServiceDirectory,
        timeout_s: float = 30.0,
    ) -> None:
        self._directory = directory
        self._timeout_s = timeout_s

    @override
    async def send(
        self,
        contribution: PartitionInvocationContribution,
    ) -> None:
        worker_instance = await self._directory.get_worker_instance_by_worker_id(
            contribution.worker_id
        )

        interface = self._select_arrow_interface(worker_instance)

        await asyncio.to_thread(
            self._send_sync,
            interface,
            contribution,
        )

    @staticmethod
    def _select_arrow_interface(
        worker_instance: WorkerInstance,
    ) -> CapabilityInterface:
        interfaces = worker_instance.get_capability_interfaces_by_type_and_protocol(
            CapabilityType.PARTITION_INFERENCE,
            CapabilityProtocol.ARROW,
        )

        if not interfaces:
            raise ValueError(
                f"Worker {worker_instance.service_id} "
                "does not expose an Arrow inference interface"
            )

        return interfaces[0]

    def _send_sync(
        self,
        interface: CapabilityInterface,
        contribution: PartitionInvocationContribution,
    ) -> None:
        batch = self._encode_contribution(contribution)

        descriptor = flight.FlightDescriptor.for_command(SUBMIT_CONTRIBUTION_COMMAND)
        options = flight.FlightCallOptions(timeout=self._timeout_s)

        endpoint = interface.endpoint.unicode_string()

        with closing(flight.connect(endpoint)) as client:
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
            raise RuntimeError("Missing partition invocation acknowledgement")

        if extra_ack is not None:
            raise RuntimeError("Received multiple acknowledgements")

        ack = PartitionInvocationContributionAck.model_validate_json(
            ack_buffer.to_pybytes()
        )

        if ack.context != contribution.context:
            raise RuntimeError("Partition invocation acknowledgement context mismatch")

    @staticmethod
    def _encode_contribution(
        contribution: PartitionInvocationContribution,
    ) -> pa.RecordBatch:
        tensor_values = {
            name: np.ascontiguousarray(tensor.value)
            for name, tensor in contribution.bundle.bundle.items()
        }

        schema = build_contribution_schema(tensor_values)

        arrays: list[pa.Array] = [
            pa.array(
                [contribution.context.model_dump_json()],
                type=pa.large_string(),
            )
        ]

        arrays.extend(
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, axis=0))
            for value in tensor_values.values()
        )

        return pa.RecordBatch.from_arrays(
            arrays,
            schema=schema,
        )
