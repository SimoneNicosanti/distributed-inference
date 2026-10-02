from struct import Struct

import pyarrow as pa

import pyarrow.types

PROTOCOL_VERSION = 1


ECHO_ACTION = "network-probe.v1.echo"
UPLOAD_COMMAND = b"network-probe.v1.upload"

PAYLOAD_SCHEMA = pa.schema(
    [
        pa.field("payload", pa.binary(), nullable=False),
    ]
)

UPLOAD_ACK = Struct("!Q")
