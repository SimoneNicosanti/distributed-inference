from pydantic import BaseModel, ConfigDict

from shared.service.service import WorkerId


class NetworkConnectionDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_worker_id: WorkerId
    throughput_bytes_per_sec: float
    round_trip_time_s: float


class NetworkProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_worker_id: WorkerId
    connections: tuple[NetworkConnectionDto, ...]
