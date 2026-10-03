from pydantic import BaseModel, ConfigDict

from shared.service.service import WorkerId


class ConnectionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    throughput_bytes_per_sec: float  ## Bytes / s
    round_trip_time_s: float


class NetworkProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    connections: dict[WorkerId, ConnectionInfo]
