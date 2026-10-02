import asyncio
import time
from contextlib import closing
from statistics import median
from typing import override

import pyarrow as pa
import pyarrow.flight as flight

from directory.contracts.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.worker.profiling.general_network_probe_integration import (
    PAYLOAD_SIZE,
    PROBE_TIMEOUT_S,
)
from integration.worker.profiling.protocols.arrow_network_probe_integration import (
    ECHO_ACTION,
    PAYLOAD_SCHEMA,
    UPLOAD_ACK,
    UPLOAD_COMMAND,
)
from worker.application.ports.outbound.profiling.network.network_probe_client import (
    NetworkProbeClient,
)
from worker.domain.directory.worker_instance import WorkerInstance
from worker.domain.profiling.network_profile import ConnectionInfo


class ArrowNetworkProbeClient(NetworkProbeClient):
    @override
    async def probe_connection(self, worker_instance: WorkerInstance) -> ConnectionInfo:

        network_probe_capabilities = worker_instance.get_capability_by_type(
            CapabilityType.NETWORK_PROFILING
        )

        if len(network_probe_capabilities) == 0:
            raise ValueError("No network probe capabilities found")

        arrow_net_probe_interfaces = []
        for capability in network_probe_capabilities:
            arrow_net_probe_interfaces.extend(
                capability.get_interfaces_by_protocol(CapabilityProtocol.ARROW)
            )

        if len(arrow_net_probe_interfaces) == 0:
            raise ValueError("No Arrow capability found")

        capability_interface = arrow_net_probe_interfaces[0]

        return await asyncio.to_thread(
            self._sync_network_probe_call, capability_interface
        )

    def _sync_network_probe_call(
        self, capability_interface: CapabilityInterface
    ) -> ConnectionInfo:

        endpoint = capability_interface.endpoint.unicode_string()

        with closing(flight.connect(endpoint)) as client:
            # Optional untimed warm-up.
            self._network_probe_round_trip_time(client)

            round_trip_time_s = median(
                self._network_probe_round_trip_time(client) for _ in range(5)
            )

            payload = bytes(PAYLOAD_SIZE)

            throughput_bps = median(
                self._network_probe_throughput(client, payload) for _ in range(3)
            )

            return ConnectionInfo(
                throughput_bytes_per_sec=throughput_bps,
                round_trip_time_s=round_trip_time_s,
            )

    def _network_probe_round_trip_time(
        self,
        client: flight.FlightClient,
    ) -> float:
        payload = b"network-probe"

        started_ns = time.perf_counter_ns()
        options = flight.FlightCallOptions(timeout=PROBE_TIMEOUT_S)
        results = list(
            client.do_action(flight.Action(ECHO_ACTION, payload), options=options)
        )

        elapsed_s = (time.perf_counter_ns() - started_ns) / (1e9)

        if len(results) != 1:
            raise RuntimeError("Invalid network probe echo response")

        if results[0].body.to_pybytes() != payload:
            raise RuntimeError("Network probe echo payload mismatch")

        return elapsed_s

    def _network_probe_throughput(
        self,
        client: flight.FlightClient,
        payload: bytes,
    ) -> float:
        batch = pa.record_batch(
            [pa.array([payload], type=pa.binary())],
            schema=PAYLOAD_SCHEMA,
        )

        descriptor = flight.FlightDescriptor.for_command(UPLOAD_COMMAND)
        options = flight.FlightCallOptions(timeout=PROBE_TIMEOUT_S)

        started_ns = time.perf_counter_ns()

        writer, metadata_reader = client.do_put(
            descriptor,
            PAYLOAD_SCHEMA,
            options=options,
        )

        acknowledgement = None
        with closing(writer) as writer:
            writer.write_batch(batch)
            writer.done_writing()

            acknowledgement = metadata_reader.read()

        elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9

        if acknowledgement is None:
            raise RuntimeError("Missing network probe acknowledgement")

        received_bytes = UPLOAD_ACK.unpack(acknowledgement.to_pybytes())[0]

        if received_bytes != len(payload):
            raise RuntimeError(
                f"Server received {received_bytes} bytes; expected {len(payload)}"
            )

        return received_bytes / elapsed_s
