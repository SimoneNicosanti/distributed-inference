from collections.abc import Iterable

import pyarrow as pa
import pyarrow.flight as flight

from integration.worker.profiling.protocols.arrow_network_probe_integration import (
    ECHO_ACTION,
    PAYLOAD_SCHEMA,
    UPLOAD_ACK,
    UPLOAD_COMMAND,
)
from integration.worker.profiling.general_network_probe_integration import PAYLOAD_SIZE


class ArrowNetworkProbeServer(flight.FlightServerBase):
    def __init__(self, location: str):
        super().__init__(location)
        pass

    def do_action(
        self,
        context: flight.ServerCallContext,
        action: flight.Action,
    ) -> Iterable[flight.Result]:
        if action.type != ECHO_ACTION:
            raise pa.ArrowInvalid(f"Unsupported action: {action.type}")

        yield flight.Result(action.body)

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightStreamReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Expected command descriptor")

        if descriptor.command != UPLOAD_COMMAND:
            raise pa.ArrowInvalid("Unsupported upload command")

        if reader.schema != PAYLOAD_SCHEMA:
            raise pa.ArrowInvalid("Invalid payload schema")

        received_bytes = 0

        for chunk in reader:
            batch = chunk.data

            if batch is None:
                continue

            payload_column = batch.column(0)
            received_bytes += payload_column.total_values_length

            if received_bytes > PAYLOAD_SIZE:
                raise pa.ArrowInvalid("Probe payload too large")

        writer.write(pa.py_buffer(UPLOAD_ACK.pack(received_bytes)))
