from pydantic import BaseModel, ConfigDict

from shared.service.service import WorkerId


class ConnectionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    throughput_bytes_per_sec: float  ## Bytes / s
    round_trip_time_s: float


class WorkerConnection(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    connection_info: ConnectionInfo


class NetworkProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    connections: list[WorkerConnection]
